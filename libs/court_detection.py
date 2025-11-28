from typing import Any
import cv2
import sys
import numpy as np
import json

# Handle both relative import (when used as module) and absolute import (when run directly)
try:
    from .court_model import reference_points, reference_lines
except ImportError:
    from court_model import reference_points, reference_lines

window_select_corners = 'Select corners'
window_show_court_lines = 'Detected court lines'
selected_corners = {}

def find_court_lines(corners):
    actual_points = []
    model_points = []

    for key in corners:
        actual_points.append(corners[key])
        model_points.append(reference_points[key])
    
    actual_points_np = np.array(actual_points, dtype=np.float32).reshape(-1, 1, 2)
    model_points_np = np.array(model_points, dtype=np.float32).reshape(-1, 1, 2)

    try:
        H, _ = cv2.findHomography(actual_points_np, model_points_np, cv2.RANSAC, 5.0)
        if H is None:
            raise Exception("Could not calculate homography matrix from provided corners. Points might be collinear or insufficient.")
        # calculate the inverse homography matrix
        H_inv = np.linalg.inv(H)
    except cv2.error as e:
        raise Exception(f"OpenCV error during homography calculation: {e}")
    except np.linalg.LinAlgError as e:
        raise Exception(f"Linear algebra error (e.g., singular matrix) during inverse homography calculation: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error during homography calculation: {e}")

    court_lines_actual = {}
    for line_name, line_coords_model in reference_lines.items():
        # convert list of lists to NumPy array for perspectiveTransform
        line_coords_model_np = np.array(line_coords_model, dtype=np.float32).reshape(-1, 1, 2)

        # apply inverse homography to get coords in actual(video) plane
        line_coords_actual_np = cv2.perspectiveTransform(line_coords_model_np, H_inv)

        # convert back to list of lists for JSON response and round to 2 decimal places
        court_lines_actual[line_name] = np.rint(line_coords_actual_np.reshape(-1, 2)).astype(int).tolist()

    return court_lines_actual


def find_corner(image, point_coord, radius=15, quality=0.8, distance=2):
    x_center, y_center = point_coord

    # Define bounding box around the point
    x1 = max(x_center - radius, 0)
    y1 = max(y_center - radius, 0)
    x2 = min(x_center + radius, image.shape[1])
    y2 = min(y_center + radius, image.shape[0])

    # Crop the Image (ROI)
    roi = image[y1:y2, x1:x2]

    # Check if the ROI is empty
    if roi.size == 0:
        return None

    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    corners_in_roi = cv2.goodFeaturesToTrack(
        gray, 
        maxCorners=1, 
        qualityLevel=quality, 
        minDistance=distance, 
        blockSize=3
    )

    if corners_in_roi is None:
        return None

    # Get the coordinates of the first (and only) corner found in the ROI
    # The corner coordinates are relative to the ROI's top-left corner (0, 0)
    corner_roi_x, corner_roi_y = corners_in_roi[0].ravel().astype(int)

    # Map the Corner Back to Original Image Coordinates
    original_corner_x = x1 + corner_roi_x
    original_corner_y = y1 + corner_roi_y

    return (int(original_corner_x), int(original_corner_y))

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at ({x}, {y})")

        label_text = input('Please enter label for this corner: ')

        corner = find_corner(param, (x,y))
        print(f"Point {label_text}: ({corner[0]}, {corner[1]})")
        selected_corners[label_text] = corner

        # draw the new corner on the image
        img_copy = param
        # draw selected point
        cv2.circle(img_copy, (x, y), 2, (0, 0, 255), -1)
        # draw corner
        cv2.circle(img_copy, corner, 2, (0, 255, 255), -1)

        # Update the display with the new corner
        cv2.imshow(window_select_corners, img_copy)

def select_corners_from_image(image):
    img_with_corners = image.copy()

    # create a window to select the corners
    cv2.namedWindow(window_select_corners)

    # set the mouse callback to the window
    cv2.setMouseCallback(window_select_corners, mouse_callback, img_with_corners)

    # display the image and wait for the interaction
    cv2.imshow(window_select_corners, img_with_corners)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27: # ESC key to break the loop
            break
    
    # cleanup
    cv2.destroyAllWindows()

def draw_court_lines(image, court_lines, color=(0, 0, 255), line_width=1, font_scale=1):
    img_with_court_lines = image.copy()
    cv2.namedWindow(window_show_court_lines)
    for line_name, line_coords in court_lines.items():
        cv2.line(
            img_with_court_lines, 
            tuple(line_coords[0]), 
            tuple(line_coords[1]), 
            color, 
            line_width
        )
        cv2.putText(
            img_with_court_lines, line_name, 
            tuple(line_coords[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            line_width
        )

    cv2.imshow(window_show_court_lines, img_with_court_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_court(image, select_corners=False, corners=None):
    global selected_corners
    
    if(select_corners):
        select_corners_from_image(image)
        # Use the global selected_corners that was modified by select_corners_from_image
        corners_dict = selected_corners.copy()
    else:
        # Parse corners from JSON string
        corners_dict = json.loads(corners) if isinstance(corners, str) else corners
        
    court_lines = find_court_lines(corners_dict)

    
    draw_court_lines(image, court_lines, line_width=2)
    
    return (court_lines, corners_dict)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python court_detection.py <filename> [select_corners] [corners]")
        sys.exit(1)
    filename = sys.argv[1]

    select_corners = False
    corners = None

    if len(sys.argv) > 2:
        select_corners = True if sys.argv[2].lower() in ["true", "1", "yes"] else False

    if not select_corners:
        if len(sys.argv) < 4:
            print("Error: When select_corners is False, you must provide [corners] file path (as 3rd argument)")
            sys.exit(1)
        corners = sys.argv[3]

    print(f"Received file name: {filename}")
    image = cv2.imread(filename)
    court_lines, court_corners = detect_court(image, select_corners, corners)
    print(court_lines)
    print(court_corners)

