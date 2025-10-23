from tracknetv3.utils.general import generate_frames
import cv2
from datetime import datetime
from tracknetv3.predict import track_ball_position
import numpy as np
from libs.court_detection import refine_corners, court_homography


video_file = 'fullcourt30-2.mp4'
test_court_coord={}

user_points = []
refined_corners = []

def click_event(event, x, y, flags, param):
    global refined_corners
    if event == cv2.EVENT_LBUTTONDOWN:
        user_points.append((x, y))
        cv2.circle(display, (x, y), 2, (255, 0, 0), -1)
        cv2.imshow("Select corners", display)

        # Once 4 corners are selected, refine them
        if len(user_points) == 4:
            refined_corners = refine_corners(display, user_points, use_canny=True)
            # Draw labels on original frame
            for i, c in enumerate(refined_corners):
                cv2.putText(display, f"C{i+1}", (int(c[0]) + 5, int(c[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.circle(display, c, 2, (0, 0, 255), -1)
            cv2.imshow("Select corners", display)

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
gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
cv2.imshow("Select corners", display)
cv2.setMouseCallback("Select corners", click_event)


cv2.waitKey(0)
cv2.destroyAllWindows()

print(refined_corners)

test_court_coord = court_homography({
    "p13": refined_corners[0],
    "p15": refined_corners[1],
    "p5": refined_corners[2],  
    "p8": refined_corners[3] 
})

print(test_court_coord["court_lines_video"])


track_ball_position(
    frame_list,
    width,
    height,
    filename,
    batch_size=8,
    court_coord=test_court_coord["court_lines_video"],
    court_corners='court_corners',
)