from tracknetv3.utils.general import generate_frames
import cv2
from datetime import datetime
from tracknetv3.predict import track_ball_position
import numpy as np


search_radius = 10  # px radius around click to search for corner
corner_quality = 0.01
min_corner_distance = 2
use_canny = True  # ✅ Toggle this to enable/disable Canny edge detection


video_file = 'halfcourt60.mp4'
test_court_coord={}

user_points = []
refined_corners = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        user_points.append((x, y))
        cv2.circle(display, (x, y), 3, (255, 0, 0), -1)
        cv2.imshow("Select corners", display)

        # Once 4 corners are selected, refine them
        if len(user_points) == 4:
            refine_corners()

def refine_corners():
    global refined_corners

    gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)

    # --- Optionally apply Canny edge detection ---
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
            cv2.circle(display, nearest_corner, 3, (0, 0, 255), -1)
        else:
            refined_corners.append((x, y))  # fallback

    # Draw labels on original frame
    for i, c in enumerate(refined_corners):
        cv2.putText(display, f"C{i+1}", (int(c[0]) + 5, int(c[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Detected Corners", display)
    print("Refined corners:", refined_corners)


cap = cv2.VideoCapture(video_file)




width, height = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# w_scaler, h_scaler = width / WIDTH, height / HEIGHT
# img_scaler = (w_scaler, h_scaler)

frame_list = generate_frames(video_file)
filename = datetime.now().strftime("%m_%d_%Y_%H:%M:%S")

# for i in range(625, len(frame_list), 1):
#     cv2.imshow(f"output_frames/frame_{i}.jpg", frame_list[i])
#     cv2.waitKey(0)
display = frame_list[0].copy()
cv2.imshow("Badminton Court Frame", frame_list[0])
cv2.setMouseCallback("Select corners", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

track_ball_position(
    frame_list,
    width,
    height,
    filename,
    batch_size=8,
    court_coord=test_court_coord,
    court_corners='court_corners',
)