import cv2
import numpy as np
from flask import Flask, request, jsonify
import base64
import json

# Initialize the Flask application
app = Flask(__name__)


# --- Helper Function to Parse Points ---
def parse_points(points_data):
    """
    Parses point data (either a string or a list) into a NumPy array of float32.

    Args:
        points_data (str or list): A JSON-encoded string of points or a direct list of lists.

    Returns:
        np.ndarray: A NumPy array of shape (N, 1, 2) with float32 points,
                    or None if parsing fails.
    """
    try:
        if isinstance(points_data, str):
            points_list = json.loads(points_data)
        elif isinstance(points_data, list):
            points_list = points_data
        else:
            print(
                f"Error: Expected points_data to be a string or list, got {type(points_data)}"
            )
            return None

        # Validate that it's a list of lists and each inner list has 2 elements
        if not isinstance(points_list, list):
            print(f"Error: Expected a list, got {type(points_list)}")
            return None
        for p in points_list:
            if not (
                isinstance(p, list)
                and len(p) == 2
                and all(isinstance(coord, (int, float)) for coord in p)
            ):
                print(f"Error: Each point must be a list of two numbers. Found: {p}")
                return None

        # Convert the list of lists to a NumPy array of float32
        # Reshape to (N, 1, 2) as required by cv2.findHomography or cv2.perspectiveTransform
        return np.array(points_list, dtype=np.float32).reshape(-1, 1, 2)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON string: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during point parsing: {e}")
        return None


# --- Function to Generate Badminton Court Lines in Destination Space ---
def generate_badminton_court_lines_in_dst_space(width, height):
    """
    Generates the coordinates for all standard badminton court lines
    within a rectangular destination space of given width and height.

    Assumes the court is oriented such that its length is along the Y-axis
    and width along the X-axis, starting from (0,0).

    Standard badminton court dimensions (approximate ratios):
    Full length: 13.4m
    Full width (doubles): 6.1m
    Singles width: 5.18m (half court width is 3.05m)
    Short service line from net: 1.98m
    Long service line (doubles) from back: 0.76m
    Long service line (singles) from back: 0.76m (but court is narrower)
    Net to center line: 6.7m

    We'll use pixel ratios based on these dimensions.
    """
    lines = {}

    # Convert meters to pixels based on total court height/width
    # Assuming court_height = 1340 pixels corresponds to 13.4m
    # Assuming court_width = 610 pixels corresponds to 6.1m

    # Ratios (approximate based on standard dimensions)
    # Total length: 13.4m
    # Total width: 6.1m

    # Short service line (from top/net side)
    # 1.98m from net, so (1.98 / 13.4) * height
    short_service_y = (1.98 / 13.4) * height
    net_y = height / 2
    lines["short_service_line_top"] = [
        [0, net_y - short_service_y],
        [width, net_y - short_service_y],
    ]
    lines["short_service_line_bottom"] = [
        [0, net_y + short_service_y],
        [width, net_y + short_service_y],
    ]

    # Long service line (doubles) from back boundary
    # 0.76m from back, so (0.76 / 13.4) * height from bottom
    long_service_y_from_bottom = (0.76 / 13.4) * height
    lines["long_service_line_top"] = [
        [0, long_service_y_from_bottom],
        [width, long_service_y_from_bottom],
    ]  # This would be the back boundary of the other side
    lines["long_service_line_bottom"] = [
        [0, height - long_service_y_from_bottom],
        [width, height - long_service_y_from_bottom],
    ]

    # Center line (divides court into left/right service boxes)
    # Half width is 3.05m (for doubles)
    center_x = width / 2.0
    lines["center_line_top"] = [
        [center_x, 0],
        [center_x, net_y - short_service_y],
    ]  # Only between short service lines
    lines["center_line_bottom"] = [
        [center_x, height],
        [center_x, net_y + short_service_y],
    ]  # Only between short service lines

    # Singles side lines (inner lines)
    # Singles width is 5.18m. So, (5.18 / 6.1) * width
    # Each side is (6.1 - 5.18) / 2 = 0.46m from the doubles sideline
    singles_offset_x = (0.46 / 6.1) * width
    lines["singles_left_line"] = [[singles_offset_x, 0], [singles_offset_x, height]]
    lines["singles_right_line"] = [
        [width - singles_offset_x, 0],
        [width - singles_offset_x, height],
    ]

    # Outer boundary lines (main court outline)
    lines["outer_boundary_top"] = [[0, 0], [width, 0]]
    lines["outer_boundary_bottom"] = [[0, height], [width, height]]
    lines["outer_boundary_left"] = [[0, 0], [0, height]]
    lines["outer_boundary_right"] = [[width, 0], [width, height]]

    # Convert all points to the required NumPy array format (N, 1, 2)
    # And group them for easier consumption by the client
    all_generated_points = {}
    for line_name, line_coords in lines.items():
        # Each line is defined by two points [start_x, start_y], [end_x, end_y]
        # We want to return them as a list of lists of points, where each inner list
        # represents a segment.
        all_generated_points[line_name] = (
            np.array(line_coords, dtype=np.float32).reshape(-1, 2).tolist()
        )

    return all_generated_points


# --- Homography Transformation Route ---
@app.route("/transform_badminton_court", methods=["POST"])
def transform_badminton_court():
    """
    Receives 4 source corner points and optionally an image (base64 encoded).
    Returns the coordinates of all badminton court lines generated in the
    destination space (bird's-eye view) AND transformed back into the
    original source image plane.

    Expects a JSON payload with 'src_points' (list of 4 [x,y] points)
    and optionally 'image_data' (base64 string of the image).
    Example:
    {
        "src_points": [[100,50],[500,80],[550,400],[80,380]],
        "image_data": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDA..." (optional)
    }

    Returns:
        JSON response containing:
        - generated_court_lines_dst: Coordinates in the bird's-eye view.
        - generated_court_lines_src: Coordinates transformed back to the source image plane.
        - output_court_dimensions: Dimensions of the bird's-eye view court.
        - homography_matrix: The calculated homography matrix.
        - inverse_homography_matrix: The calculated inverse homography matrix.
    """
    data = request.get_json()
    if not data or "src_points" not in data:
        return (
            jsonify(
                {
                    "error": "Missing 'src_points' parameter in request body. Please provide a list of 4 [x,y] points."
                }
            ),
            400,
        )

    src_points_data = data["src_points"]
    image_b64 = data.get("image_data")  # image_data is now optional

    # Parse the source points
    src_court_corners = parse_points(src_points_data)

    if src_court_corners is None or src_court_corners.shape[0] != 4:
        return (
            jsonify(
                {
                    "error": "Invalid 'src_points' format or incorrect number of points. Expected a list of 4 [x,y] points."
                }
            ),
            400,
        )

    # --- Image Decoding (optional, only if image_data is provided) ---
    image = None
    if image_b64:
        try:
            image_bytes = base64.b64decode(image_b64)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image is None:
                print(
                    "Warning: Could not decode optional image data. Proceeding without image."
                )

        except Exception as e:
            print(
                f"Warning: Error decoding optional image: {e}. Proceeding without image."
            )

    # --- 1. Define Destination Points for the Full Court ---
    # Standard badminton court dimensions (example values in pixels)
    # A full badminton court is approx 13.4m x 6.1m (doubles)
    # We'll use a proportional pixel representation.
    # Let's define a court with width 610 pixels and height 1340 pixels for example.
    court_width = 610
    court_height = 1340

    dst_court_corners = np.array(
        [
            [0, court_height / 2],  # Top-left
            [court_width, court_height / 2],  # Top-right
            [court_width, court_height],  # Bottom-right
            [0, court_height],  # Bottom-left
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    # --- 2. Calculate Homography ---
    try:
        H, _ = cv2.findHomography(src_court_corners, dst_court_corners, cv2.RANSAC, 5.0)
        if H is None:
            return (
                jsonify(
                    {
                        "error": "Could not calculate homography matrix from provided corners. Points might be collinear or insufficient."
                    }
                ),
                500,
            )

        # Calculate the inverse homography matrix
        H_inv = np.linalg.inv(H)

    except cv2.error as e:
        return (
            jsonify({"error": f"OpenCV error during homography calculation: {e}"}),
            500,
        )
    except np.linalg.LinAlgError as e:
        return (
            jsonify(
                {
                    "error": f"Linear algebra error (e.g., singular matrix) during inverse homography calculation: {e}"
                }
            ),
            500,
        )
    except Exception as e:
        return (
            jsonify(
                {
                    "error": f"An unexpected error occurred during homography calculation: {e}"
                }
            ),
            500,
        )

    # --- 3. Generate All Court Lines in the Destination Space (Bird's-Eye View) ---
    generated_court_lines_dst = generate_badminton_court_lines_in_dst_space(
        court_width, court_height
    )

    # --- 4. Transform Generated Lines Back to Source Plane ---
    transformed_src_lines_coords = {}
    for line_name, line_coords_dst in generated_court_lines_dst.items():
        # Convert list of lists to NumPy array for perspectiveTransform
        line_points_dst_np = np.array(line_coords_dst, dtype=np.float32).reshape(
            -1, 1, 2
        )

        # Apply inverse homography to get points in source plane
        line_points_src_np = cv2.perspectiveTransform(line_points_dst_np, H_inv)

        # Convert back to list of lists for JSON response and round to 2 decimal places
        transformed_src_lines_coords[line_name] = np.round(
            line_points_src_np.reshape(-1, 2), 2
        ).tolist()

    # Round generated_court_lines_dst coordinates to 2 decimal places as well
    rounded_generated_court_lines_dst = {}
    for line_name, line_coords_dst in generated_court_lines_dst.items():
        rounded_generated_court_lines_dst[line_name] = np.round(
            np.array(line_coords_dst), 2
        ).tolist()

    # --- 5. Return Generated Coordinates ---
    return (
        jsonify(
            {
                "message": "Badminton court line coordinates generated and transformed.",
                "generated_court_lines_dst": rounded_generated_court_lines_dst,
                "generated_court_lines_src": transformed_src_lines_coords,
                "output_court_dimensions": {
                    "width": court_width,
                    "height": court_height,
                },
                "homography_matrix": np.round(H, 2).tolist(),
                "inverse_homography_matrix": np.round(H_inv, 2).tolist(),
            }
        ),
        200,
    )


# --- Main execution block ---
if __name__ == "__main__":
    # Run the Flask app in debug mode (for development)
    # In a production environment, use a production-ready WSGI server like Gunicorn.
    app.run(debug=True, host="0.0.0.0", port=6000)
