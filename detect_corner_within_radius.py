import cv2
import numpy as np

# --- Parameters ---
video_path = "halfcourt30.mp4"  # your video file
search_radius = 10  # px radius around click to search for corner
corner_quality = 0.8
min_corner_distance = 2
use_canny = False  # ✅ Toggle this to enable/disable Canny edge detection

# --- Load first frame from video ---
cap = cv2.VideoCapture(video_path)
ret, img = cap.read()
cap.release()

if not ret:
    raise ValueError("❌ Could not read first frame from video.")

display = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Storage for user clicks and refined corners ---
user_points = []
refined_corners = []

# --- Mouse callback for user clicks ---
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

# --- Run ---
cv2.imshow("Select corners", display)
cv2.setMouseCallback("Select corners", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
