import subprocess
import os
import cv2
from collections import deque
from PIL import Image, ImageDraw
import numpy as np

def draw_court_lines(image, court_lines, color=(0, 0, 255), line_width=1, font_scale=1):
    img_with_court_lines = image.copy()
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
def draw_traj(img, traj, radius=3, color="red"):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    for i in range(len(traj)):
        if traj[i] is not None:
            draw_x = traj[i][0]
            draw_y = traj[i][1]
            bbox = (draw_x - radius, draw_y - radius, draw_x + radius, draw_y + radius)
            draw.ellipse(bbox, fill="rgb(255,255,255)", outline=color)
    del draw
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img

def reencode_video(input_file, output_file):
    # Construct the FFmpeg command
    command = [
        'ffmpeg',       # FFmpeg executable
        '-loglevel', 'quiet',  # Suppress all output
        '-i', input_file,  # Input file
        '-movflags', 'faststart',
        '-y',  # Overwrite output file if it exists
        output_file      # Output file
    ]
    
    try:
        # Run the command with output suppressed
        subprocess.run(
            command, 
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if os.path.exists(input_file):
            os.remove(input_file)
        else:
            print(f"{input_file} does not exist.")

    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")


def draw_ground_hit_point(frame, frame_i, decision_df):
    # circle setting
    circle_radius = 5
    out_color = (0, 0, 255)
    in_color = (0, 255, 0)
    circle_thickness = 5

    # text setting
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    text_color = (0, 0, 255)
    text_thickness = 2

    for row in decision_df.itertuples(index=True, name="Frame"):
        if row.Frame <= frame_i:
            color = in_color if row.decision == 'IN' else out_clor

            circle_center = (int(row.X),int(row.Y))  # Center of the frame  # Center of the frame
            cv2.circle(frame, circle_center, circle_radius, color, circle_thickness)

            text = row.decision

            # Position the text slightly below and to the right of the circle
            text_position = (
                circle_center[0] + circle_radius + 5,
                circle_center[1] + circle_radius - 15,
            )

            # Add the text to the frame
            cv2.putText(
                frame,
                text,
                text_position,
                font,
                font_scale,
                color,
                text_thickness,
                cv2.LINE_AA,
            )

    return frame

def generate_result_video(
    frame_list,
    video_width,
    video_height,
    pred_dict,
    decision,
    save_file_path,
    fps=60,
    traj_len=8,
):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Read prediction result
    x_pred, y_pred, vis_pred = pred_dict["X"], pred_dict["Y"], pred_dict["Visibility"]
    
    # Video config
    out = cv2.VideoWriter(save_file_path, fourcc, fps, (video_width, video_height))
    out_original = cv2.VideoWriter(save_file_path.replace('.mp4', '_original_temp.mp4'), fourcc, fps, (video_width, video_height))

    # Create a queue for storing trajectory
    pred_queue = deque()

    # Draw label and prediction trajectory
    for i, frame in enumerate(frame_list):

        out_original.write(frame)
        # Check capacity of queue
        if len(pred_queue) >= traj_len:
            pred_queue.pop()

        # Push ball coordinates for each frame
        (
            pred_queue.appendleft([x_pred[i], y_pred[i]])
            if vis_pred[i]
            else pred_queue.appendleft(None)
        )
        # Draw prediction trajectory
        frame = draw_traj(frame, pred_queue, color="yellow")

        frame = draw_ground_hit_point(frame, i, decision)

        # Position the text slightly below and to the right of the circle
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        text_color = (0, 0, 255) 
        text_thickness = 2
        text_position = (80,80)

        # Add the text to the frame
        cv2.putText(
            frame,
            f'Frame: {i}',
            text_position,
            font,
            font_scale,
            text_color,
            text_thickness,
            cv2.LINE_AA,
        )

        out.write(frame)
        i += 1
    out.release()
    out_original.release()
    reencode_video(save_file_path, save_file_path.replace(".mp4", "_result.mp4"))
    reencode_video(save_file_path.replace('.mp4', '_original_temp.mp4'), save_file_path.replace(".mp4", "_original.mp4"))
