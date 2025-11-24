import cv2
import numpy as np
from flask import Flask, request, jsonify
import base64
import json
from libs.court_detection import detect_court, find_corner
from libs.court_model import court_width, court_height, reference_points

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





@app.route("/", methods=["GET"])
def get():
    return {"message": "Hello"}, 200


# --- Homography Transformation Route ---
@app.route("/transform_badminton_court", methods=["POST"])
def transform_badminton_court():
    """
    Receives at least 4 source points.
    Returns the coordinates of all badminton court lines generated in the
    destination space (bird's-eye view) AND transformed back into the
    original source image plane.

    Expects a JSON payload with 'src_points'
    Example:
    {
        "src_points": { 
            "p1":[464,490],
            "p2": [824,493],
            "p3": [1192,667],
            "p4": [112,662]
        }
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
            jsonify({"error": "Missing 'src_points' parameter in request body. Please provide a list of 4 [x,y] points."}),
            400,
        )
    # Receive 'image_frame' as base64 in addition to 'src_points'
    # The expected payload now includes:
    # {
    #   "src_points": {...},
    #   "image_frame": "..."  # base64-encoded image string
    # }
    if "image_frame" not in data:
        return (
            jsonify({"error": "Missing 'image_frame' parameter in request body. Please provide a base64-encoded image."}),
            400,
        )
    
    image_base64 = data["image_frame"]
    src_points_data = data["src_points"]
    


    # --- 2. Calculate Homography ---
    try:
        # Decode the base64 image string to bytes
        image_bytes = base64.b64decode(image_base64)
        # Convert bytes to a numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode the numpy array to an OpenCV image (BGR format)
        image_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # If the image failed to decode, raise an error
        if image_frame is None:
            return (
                jsonify({"error": "Failed to decode the provided base64 image."}),
                400,
            )

        # find corners from input data and image
        court_intersection = {}
        for key, value in src_points_data.items():
            print('key', key)
            # Ensure value[0] and value[1] are integers before using as coordinates
            x = int(value[0])
            y = int(value[1])
            print(x, y)
            court_intersection[key] = find_corner(image_frame, (x, y))

        print(court_intersection)
        court_lines, corners_dict = detect_court(image_frame, corners=court_intersection)
        print(court_lines)
        return (jsonify(court_lines), 200)

    except Exception as e:
        print(e)
        return (
            jsonify({"error": f"{e}"}),
            500,
        )

    


# --- Main execution block ---
if __name__ == "__main__":
    # Run the Flask app in debug mode (for development)
    # In a production environment, use a production-ready WSGI server like Gunicorn.
    app.run(debug=True, host="0.0.0.0", port=8080)
