import sys
from pathlib import Path
import argparse
import pandas as pd
import time

# Add parent directory to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
from datetime import datetime
from tracknetv3.predict import track_ball_position
import numpy as np
from libs.court_detection import detect_court, draw_court_lines
from libs.generate_result_video import generate_result_video
from libs.process_data import mark_default_shuttlecock_position
from libs.line_decision import line_judge_decision

def format_time(seconds):
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} minute(s) {secs:.2f} seconds"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours} hour(s) {minutes} minute(s) {secs:.2f} seconds"

def generate_frames(video_file):
    """Sample frames from the video.

    Args:
        video_file (str): File path of the video file

    Returns:
        frame_list (List[numpy.ndarray]): List of sampled frames
    """

    assert video_file[-4:] == ".mp4", "Invalid video file format."

    # Get camera parameters
    cap = cv2.VideoCapture(video_file)
    frame_list = []
    success = True

    # Sample frames until video end
    while success:
        success, frame = cap.read()
        if success:
            frame_list.append(frame)

    return frame_list

def main():
    total_start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Process a video file and predicted results CSV to visualize court and predictions.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file (.mp4)")
    parser.add_argument("--pred_dict", type=str, required=False, help="Path to prediction CSV file")
    args = parser.parse_args()

    print("=" * 60)
    print("Starting video processing pipeline...")
    print("=" * 60)
    
    # Load video frames
    print("\n[1/5] Loading video frames...")
    section_start = time.time()
    cap = cv2.VideoCapture(args.video)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_list = generate_frames(args.video)
    filename = f'{datetime.now().strftime("%d_%m_%Y_%H:%M:%S")}'
    section_time = time.time() - section_start
    print(f"✓ Loaded {len(frame_list)} frames ({video_width}x{video_height}) - {format_time(section_time)}")

    # Detect court lines
    print("\n[2/5] Detecting court lines...")
    section_start = time.time()
    court_frame = frame_list[0].copy()
    court_lines, court_corners = detect_court(court_frame, True)
    section_time = time.time() - section_start
    print(f"✓ Court detection completed - {format_time(section_time)}")

    # Ball tracking or loading predictions
    print("\n[3/5] Processing ball tracking/predictions...")
    section_start = time.time()
    if args.pred_dict is not None:
        # Load predicted results CSV
        pred_df = pd.read_csv(args.pred_dict)
        pred_dict = {
            "Frame": pred_df["Frame"].tolist() if "Frame" in pred_df else list(range(len(frame_list))),
            "X": pred_df["X"].tolist(),
            "Y": pred_df["Y"].tolist(),
            "Visibility": pred_df["Visibility"].tolist() if "Visibility" in pred_df else [1]*len(pred_df),
        }
        print(f"✓ Loaded predictions from CSV - {len(pred_df)} entries")
    else:
        print("  Running ball tracking (this may take a while)...")
        pred_dict = track_ball_position(
            frame_list,
            video_width,
            video_height,
            filename,
            batch_size=8,
        )
        print(f"✓ Ball tracking completed - {len(pred_dict['Frame'])} predictions")
    section_time = time.time() - section_start
    print(f"✓ Prediction processing completed - {format_time(section_time)}")

    # Line judge decision
    print("\n[4/5] Making line judge decisions...")
    section_start = time.time()
    pred_dict = mark_default_shuttlecock_position(pred_dict)
    decision_df = line_judge_decision(pred_dict, court_lines, f"prediction/{filename}_plot.png", 69)
    section_time = time.time() - section_start
    print(f"✓ Line judge decisions completed - {format_time(section_time)}")
    if not decision_df.empty:
        print(f"  Found {len(decision_df)} ground hit candidate(s)")

    # Generate result video
    print("\n[5/5] Generating result video...")
    section_start = time.time()
    file_path = f"prediction/{filename}.mp4"
    generate_result_video(
        frame_list,
        video_width,
        video_height,
        pred_dict,
        decision_df,
        court_lines,
        file_path,
        fps=fps
    )
    section_time = time.time() - section_start
    print(f"✓ Result video generated - {format_time(section_time)}")
    print(f"  Saved to: {file_path}")
    
    # Print total time
    total_time = time.time() - total_start_time
    print("\n" + "=" * 60)
    print(f"✓ Pipeline completed successfully!")
    print(f"  Total execution time: {format_time(total_time)}")
    print("=" * 60)

if __name__ == "__main__":
    main()


