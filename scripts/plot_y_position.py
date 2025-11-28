import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot Y position over frames from a CSV file.")
    parser.add_argument("csv_path", type=str, help="Path to the prediction CSV file")
    parser.add_argument("--ground_hit_frame", type=int, default=None, 
                        help="Frame number where the ball hits the ground (optional)")
    args = parser.parse_args()

    # Read CSV file
    df = pd.read_csv(args.csv_path)

    # Only plot visible points
    if "Visibility" in df.columns:
        mask = df["Visibility"] == 1
    else:
        mask = [True] * len(df)

    frames = df["Frame"][mask]
    y_positions = df["Y"][mask]

    plt.figure(figsize=(12, 6))
    plt.plot(frames, y_positions, marker='o', linestyle='-', color='b')

    # Mark ground hit frame if provided
    if hasattr(args, "ground_hit_frame") and args.ground_hit_frame is not None:
        ghf = int(args.ground_hit_frame)
        # Only mark if it exists in frames
        if ghf in frames.values:
            y_ghf = y_positions[frames == ghf].values[0]
            plt.scatter([ghf], [y_ghf], color='red', zorder=5, label='Ground Hit Frame')
            plt.axvline(x=ghf, color='red', linestyle='--', alpha=0.7)
        else:
            print(f"Warning: ground_hit_frame ({ghf}) not found in visible frames.")

    plt.xlabel("Frame")
    plt.ylabel("Y Position")
    plt.title(f"Y Position Over Frames: {args.csv_path}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_path = args.csv_path.rsplit(".", 1)[0] + "_y_position_plot.png"
    plt.savefig(output_path)

if __name__ == "__main__":
    main()
