import cv2
import numpy as np
from pathlib import Path

def load_points_from_txt(txt_file):
    points = []
    with open(txt_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and ";" in line:
                x_str, y_str = line.split(";")
                x, y = float(x_str), float(y_str)
                points.append((int(round(x)), int(round(y))))
    return points

def draw_points_on_frame(frame, points, color=(0, 0, 255), radius=8, thickness=-1):
    output = frame.copy()
    for idx, (x, y) in enumerate(points):
        cv2.circle(output, (x, y), radius, color, thickness)
        cv2.putText(
            output, 
            str(idx+1), 
            (x+10, y-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            color, 
            2, 
            cv2.LINE_AA
        )
    return output

def main():
    path_to_video = "../sample_media/full.avi"
    path_to_points = "../sample_media/outfull.txt"
    output_image = "frame_with_points_full.jpg"

    # Open video
    cap = cv2.VideoCapture(path_to_video)
    success, frame = cap.read()
    cap.release()

    if not success:
        print(f"Error: Cannot read first frame from {path_to_video}")
        return

    # Load points
    points = load_points_from_txt(path_to_points)

    # Draw
    frame_with_points = draw_points_on_frame(frame, points)

    # Draw lines connecting consecutive points
    frame_with_lines = frame.copy()
    if len(points) > 1:
        for i in range(4):
            pt1 = points[i]
            if i < 3:
                pt2 = points[i+1]
            else:
                pt2 = points[0]
            cv2.line(frame_with_lines, pt1, pt2, (0, 0, 255), 2)
            # Put coordinate text next to each point
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            color = (0, 255, 0)
            # Display coordinates for pt1
            cv2.putText(
                frame_with_lines,
                f"({pt1[0]},{pt1[1]})",
                (pt1[0]+10, pt1[1]-10),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA
            )
            # For the last segment, also display pt2's coordinates
            if i == len(points) - 2:
                cv2.putText(
                    frame_with_lines,
                    f"({pt2[0]},{pt2[1]})",
                    (pt2[0]+10, pt2[1]-10),
                    font,
                    font_scale,
                    color,
                    thickness,
                    cv2.LINE_AA
                )

    # Show and save
    cv2.imshow("Frame with Lines", frame_with_lines)
    cv2.imwrite(output_image, frame_with_lines)
    print(f"Saved frame with lines as {output_image}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
