import sys
from pathlib import Path
import argparse
import pandas as pd

# Add parent directory to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tracknetv3.utils.general import generate_frames
import cv2
from datetime import datetime
from tracknetv3.predict import track_ball_position
import numpy as np
from libs.court_detection import detect_court, draw_court_lines
from tracknetv3.utils.general import draw_court
from libs.generate_result_video import generate_result_video


# # Video file path relative to project root
# video_file = str(project_root / 'sample_media' / 'fullcourt120-2.mp4')

# cap = cv2.VideoCapture(video_file)

# video_width, video_height = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# frame_list = generate_frames(video_file)
# filename = datetime.now().strftime("%m_%d_%Y_%H:%M:%S")

# court_frame = frame_list[0].copy()
# court_lines, court_corners = detect_court(court_frame, True)

# print('Court lines')
# print('-' * 10)
# print(court_lines)
# print('Court intersections')
# print('-' * 10)
# print(court_corners)


# pred_dict = track_ball_position(
#     frame_list,
#     video_width,
#     video_height,
#     filename,
#     batch_size=8,
#     # court_coord=test_court_coord["court_lines_video"],
#     # court_corners='court_corners',
# )



def main():
    parser = argparse.ArgumentParser(description="Process a video file and predicted results CSV to visualize court and predictions.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file (.mp4)")
    parser.add_argument("--pred_dict", type=str, required=False, help="Path to prediction CSV file")
    args = parser.parse_args()

    # Load video frames
    cap = cv2.VideoCapture(args.video)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_list = generate_frames(args.video)
    filename = f'{datetime.now().strftime("%d_%m_%Y_%H:%M:%S")}'

    # # Detect court lines
    # court_frame = frame_list[0].copy()
    # court_lines, court_corners = detect_court(court_frame, True)

    # print('Court lines')
    # print('-' * 10)
    # print(court_lines)
    # print('Court intersections')
    # print('-' * 10)
    # print(court_corners)

    if args.pred_dict is not None:
        # Load predicted results CSV
        pred_df = pd.read_csv(args.pred_dict)
        pred_dict = {
            "Frame": pred_df["Frame"].tolist() if "Frame" in pred_df else list(range(len(frame_list))),
            "X": pred_df["X"].tolist(),
            "Y": pred_df["Y"].tolist(),
            "Visibility": pred_df["Visibility"].tolist() if "Visibility" in pred_df else [1]*len(pred_df),
        }
    else:
        pred_dict = track_ball_position(
            frame_list,
            video_width,
            video_height,
            filename,
            batch_size=8,
        )

    # You can add additional logic for decision, etc., as needed.
    # For demonstration, we'll pass an empty DataFrame for decision.
    decision = pd.DataFrame([])

    save_file = f"prediction/{filename}.mp4"
    generate_result_video(
        frame_list,
        video_width,
        video_height,
        pred_dict,
        save_file,
    )
    print(f"Result video saved to: {save_file}")

if __name__ == "__main__":
    main()


