import cv2
import numpy as np
from .court_model import court_width, court_height, reference_points
import json

def refine_corners(image, user_points, use_canny=True):
    search_radius = 15  # px radius around click to search for corner
    corner_quality = 0.8
    min_corner_distance = 2
    refined_corners = []

    display = image.copy()
    gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)

    if use_canny:
        print("🔍 Using Canny edge detection for refinement...")
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        proc_img = cv2.Canny(blurred, 50, 150)
    else:
        print("🎯 Using grayscale image for refinement...")
        proc_img = gray

    for (x, y) in user_points:
        x1, y1 = max(x - search_radius, 0), max(y - search_radius, 0)
        x2, y2 = min(x + search_radius, proc_img.shape[1]), min(y + search_radius, proc_img.shape[0])
        roi = proc_img[y1:y2, x1:x2]

        # Detect corners in ROI
        corners = cv2.goodFeaturesToTrack(
            roi, maxCorners=5, qualityLevel=corner_quality, minDistance=min_corner_distance
        )

        if corners is not None:
            corners = corners.astype(int)
            corners = [(c.ravel()[0] + x1, c.ravel()[1] + y1) for c in corners]
            distances = [np.hypot(cx - x, cy - y) for cx, cy in corners]
            nearest_corner = corners[np.argmin(distances)]
            refined_corners.append(nearest_corner)
        else:
            refined_corners.append((x, y))  # fallback
    print("✅ Corner refinement complete.")
    return refined_corners


def court_homography(user_points):
    video_points = []
    model_points = []
    for key in user_points:
        video_points.append(user_points[key])
        model_points.append(reference_points[key])

    video_points_np = np.array(video_points, dtype=np.float32).reshape(-1, 1, 2)
    model_points_np = np.array(model_points, dtype=np.float32).reshape(-1, 1, 2)

    try:
        H, _ = cv2.findHomography(video_points_np, model_points_np, cv2.RANSAC, 5.0)
        if H is None:
            raise Exception("Could not calculate homography matrix from provided corners. Points might be collinear or insufficient.")

        # Calculate the inverse homography matrix
        H_inv = np.linalg.inv(H)

    except cv2.error as e:
        raise Exception(f"OpenCV error during homography calculation: {e}")
    except np.linalg.LinAlgError as e:
        raise Exception(f"Linear algebra error (e.g., singular matrix) during inverse homography calculation: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error during homography calculation: {e}")
    
    court_lines_model = generate_court_lines_in_top_view(court_width, court_height)

    court_lines_video = {}
    for line_name, line_coords_dst in court_lines_model.items():
        # Convert list of lists to NumPy array for perspectiveTransform
        line_points_model_np = np.array(line_coords_dst, dtype=np.float32).reshape(-1, 1, 2)

        # Apply inverse homography to get points in video plane
        line_points_src_np = cv2.perspectiveTransform(line_points_model_np, H_inv)

        # Convert back to list of lists for JSON response and round to 2 decimal places
        court_lines_video[line_name] = np.round(line_points_src_np.reshape(-1, 2), 2).tolist()
    
    rounded_court_lines_model = {}
    for line_name, line_coords_dst in court_lines_model.items():
        rounded_court_lines_model[line_name] = np.round(
            np.array(line_coords_dst), 2
        ).tolist()

    return {
                "court_lines_model": rounded_court_lines_model,
                "court_lines_video": court_lines_video,
                "model_court_dimensions": {
                    "width": court_width,
                    "height": court_height,
                },
                "homography_matrix": np.round(H, 2).tolist(),
                "inverse_homography_matrix": np.round(H_inv, 2).tolist(),
            }


def generate_court_lines_in_top_view(width, height):
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
    
    short_service_y = (1.98 / 13.4) * height # 1.98m from net, so (1.98 / 13.4) * height
    net_y = height / 2
    line_width = (0.04 / 13.4) * height  # 0.04m line width in pixels

    lines["net_line"] = [[0, net_y], [width, net_y]]
    lines["short_service_line_top"] = [
        [0, net_y - short_service_y],
        [width, net_y - short_service_y],
    ]
    lines["short_service_line_bottom"] = [
        [0, net_y + short_service_y],
        [width, net_y + short_service_y],
    ]

    # Long service line (doubles) from back boundary
    long_service_y_from_bottom = (0.76 / 13.4) * height # 0.76m from back, so (0.76 / 13.4) * height from bottom
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
    lines["center_line_top_left"] = [
        [center_x - line_width / 2, 0],
        [center_x - line_width / 2, net_y - short_service_y],
    ]  # Only between short service lines
    lines["center_line_top_right"] = [
        [center_x + line_width / 2, 0],
        [center_x + line_width / 2, net_y - short_service_y],
    ]  # Only between short service lines
    lines["center_line_bottom_left"] = [
        [center_x - line_width / 2, height],
        [center_x - line_width / 2, net_y + short_service_y],
    ]  # Only between short service lines
    lines["center_line_bottom_right"] = [
        [center_x + line_width / 2 , height],
        [center_x + line_width / 2, net_y + short_service_y],
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
        all_generated_points[line_name] = (np.array(line_coords, dtype=np.float32).reshape(-1, 2).tolist())

    return all_generated_points