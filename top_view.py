import cv2
import numpy as np
from homography import homography
import pandas as pd
from PIL import Image, ImageDraw


def create_court_frame(
    frame_width,
    frame_height,
    ball_x,
    ball_y,
    court_color=(255, 255, 255),
    background_color=(0, 200, 0),
):
    """
    Creates a single video frame with a green background and a white badminton court.

    Args:
        frame_width (int): The width of the video frame in pixels.
        frame_height (int): The height of the video frame in pixels.
        court_color (tuple): The BGR color of the court lines (default is white).
        background_color (tuple): The BGR color of the background (default is a dark green).

    Returns:
        numpy.ndarray: The generated video frame with the court drawn on it.
    """
    # 1. Create a blank image (NumPy array) with the specified dimensions and background color.
    # The array should be of type uint8 (8-bit unsigned integer) for pixel values.
    # OpenCV uses BGR color format by default.
    frame = np.full((frame_height, frame_width, 3), background_color, dtype=np.uint8)

    # 2. Define the pixel coordinates of a standard badminton half-court.
    # These coordinates are for a top-down view and need to be scaled to your frame size.
    # Let's define the court in a normalized coordinate system (0 to 1) first for easy scaling.
    # Dimensions in meters: length=6.7, width=6.1
    # Court lines thickness is often around 4cm.
    # This is a simplified representation of a half-court.
    # The coordinates are [x, y] format, where (0,0) is top-left.

    # To place the court in the center, we need to calculate an offset.
    court_width = 400
    court_height = 800

    # # Calculate top-left corner of the court to center it
    x_offset = 0
    y_offset = 0
    # x_offset = (frame_width - court_width) // 2
    # y_offset = (frame_height - court_height) // 2

    # Define court lines relative to a top-left origin (0,0)
    # The format is a dictionary where each key is the line name
    # and the value is a list of two points [start_point, end_point].

    # [[479.4, 255.3], [864.7, 243.7], [1243.4, 435.4], [68.4, 441.5]]
    # [[465.3, 249.2], [859.2, 246.1], [1223.2, 436.0], [39.1, 442.7]]

    result = homography(
        [[465.3, 249.2], [859.2, 246.1], [1223.2, 436.0], [39.1, 442.7]]
    )

    # print(result)

    court_height = abs(
        result["generated_court_lines_src"]["outer_boundary_top"][0][1]
        - result["generated_court_lines_src"]["outer_boundary_bottom"][0][1]
    )
    court_width = abs(
        result["generated_court_lines_src"]["outer_boundary_left"][0][0]
        - result["generated_court_lines_src"]["outer_boundary_right"][0][0]
    )

    x_offset = 0
    y_offset = 0

    scale = 1
    # scale = frame_height / court_height

    print(f"court_height = {court_height}")
    print(f"frame_height = {frame_height}")
    print(f"court_width = {court_width}")
    print(f"frame_width = {frame_width}")
    print(f"scale = {scale}")

    # 3. Draw the lines on the blank frame
    line_thickness = 2
    for line_name, points in result["generated_court_lines_src"].items():
        # Apply the offset to center the court
        start_point = (
            int(points[0][0] * scale + x_offset),
            int(points[0][1] * scale + y_offset),
        )
        end_point = (
            int(points[1][0] * scale + x_offset),
            int(points[1][1] * scale + y_offset),
        )

        print(start_point)
        print(end_point)
        print("-----------")
        # Use cv2.line to draw the line
        cv2.line(frame, start_point, end_point, court_color, line_thickness)

    H = np.array(result["inverse_homography_matrix"])
    print(f"({ball_x},{ball_y})")
    src_center = np.array([[[ball_x, ball_y]]], dtype=np.float32)
    dst_center = cv2.perspectiveTransform(src_center, H)[0][0]
    cv2.circle(
        frame,
        # (int(dst_center[0]), int(dst_center[1])),
        (ball_x, ball_y),
        10,
        (255, 0, 0),
        line_thickness,
    )

    return frame


# --- Usage Example ---
if __name__ == "__main__":

    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    frame_width = 1280
    frame_height = 720

    df = pd.read_csv("prediction/test_function_ball.csv")
    frame_array = df.values

    print(frame_array)

    out = cv2.VideoWriter(
        "map_ball_with_court.mp4", fourcc, fps, (frame_width, frame_height)
    )

    for frame in frame_array:
        # Create the blank frame with the court drawn on it
        court_frame = create_court_frame(
            frame_width,
            frame_height,
            frame[2],
            frame[3],
        )
        out.write(court_frame)

    # Display the frame (optional)
    cv2.imshow("Badminton Court Frame", court_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # If you want to create a video, you can do this:
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # For MP4 format
    # out = cv2.VideoWriter('badminton_court_video.mp4', fourcc, 20.0, (frame_width, frame_height))

    # for _ in range(60): # Write 60 frames (3 seconds at 20 fps)
    #     out.write(court_frame)

    # out.release()
    # print("Video saved as badminton_court_video.mp4")
