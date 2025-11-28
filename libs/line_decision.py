import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Handle both relative import (when used as module) and absolute import (when run directly)
try:
    from .process_data import mark_default_shuttlecock_position
except ImportError:
    from process_data import mark_default_shuttlecock_position

def prepare_feature(pred_dict):
    pred_dict = mark_default_shuttlecock_position(pred_dict)
    # Remove rows where Visibility is 0
    df = pd.DataFrame(pred_dict)
    df = df[df["Visibility"] != 0].reset_index(drop=True)

    # Compute vertical velocity (vy) using finite difference (y difference between consecutive frames)
    # vy = (y[i] - y[i-1]) / (frame[i] - frame[i-1])
    df["vy"] = df["Y"].diff() / df["Frame"].diff()

    # Compute vertical acceleration (ay) as the difference of vy over difference in frames
    df["ay"] = df["vy"].diff() / df["Frame"].diff()


    return df

def find_ground_hit_frame(features, lower_bound=5, higher_bound=30):

    candidate_frames = features.loc[
        ((features['vy'] > 0) & (features['vy'].shift(-1) <= 0)) &  # falling → rising/stopping
        (abs(features['ay']).shift(-1).between(lower_bound, higher_bound))
    ]

    return candidate_frames

def find_ball_below_net_pole(ground_hit_df, net_line):
    net_pole_left = net_line[0]
    net_pole_right = net_line[1]

    # Make an explicit copy to avoid SettingWithCopyWarning
    ground_hit_df = ground_hit_df.copy()

    slope = (net_pole_right[1] - net_pole_left[1]) / (net_pole_right[0] - net_pole_left[0])
    x_pred, y_pred = ground_hit_df["X"], ground_hit_df["Y"]
    y_on_net_pole = slope * (x_pred - net_pole_left[0]) + net_pole_left[1]

    ground_hit_df['y_on_net_pole'] = y_on_net_pole

    filtered_df = ground_hit_df[ground_hit_df['Y'] > ground_hit_df['y_on_net_pole']]

    return filtered_df

def plot_ground_hit_data(features, gh_candidates, gh_below_net_pole, gh_ground_truth, save_plot): # gh = ground hit
    plt.figure(figsize=(15, 8))

    # Separate y position to one graph, ay and vy in one graph

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # Plot Y Position (only) in first graph
    ax1.plot(features["Frame"], features["Y"], label="Y Position", color="blue", marker='o', markersize=3)
    ax1.invert_yaxis()
    if not gh_candidates.empty:
        ax1.scatter(gh_candidates["Frame"], gh_candidates["Y"], color="red", zorder=5, label="Ground Hit Frame Candidate(s) (Y)")
        for f in gh_candidates["Frame"]:
            ax1.axvline(x=f, color='red', linestyle='--', alpha=0.3)

    if not gh_below_net_pole.empty:
        ax1.scatter(gh_below_net_pole["Frame"], gh_below_net_pole["Y"], color="green", zorder=5, label="Ground Hit Frames below net pole (Y)")
        for f in gh_below_net_pole["Frame"]:
            ax1.axvline(x=f, color='red', linestyle='--', alpha=0.3)
    
    if gh_ground_truth is not None:
        # Plot a vertical line at the provided ground-hit frame
        ax1.axvline(x=gh_ground_truth, color='cyan', linestyle='-', alpha=0.5, label="Ground Hit (Ground Truth)")
    ax1.set_ylabel("Y Position")
    ax1.set_title("Y Position over Frames")
    ax1.legend()
    ax1.grid(True)

    # Plot vy and ay in second graph
    ax2.plot(features["Frame"], features["vy"], label="Vertical Velocity (vy)", color="orange", marker='o', markersize=3)
    ax2.plot(features["Frame"], features["ay"], label="Vertical Acceleration (ay)", color="green", marker='o', markersize=3)
    if not gh_candidates.empty:
        for idx, f in enumerate(gh_candidates["Frame"]):
            # Only add label on first iteration to avoid duplicate labels in legend
            label = 'Ground Hit Frame Candidate(s) ' if idx == 0 else ''
            ax2.axvline(x=f, color='red', linestyle='--', label=label, alpha=0.3)
    if gh_ground_truth is not None:
        # Plot a vertical line at the provided ground-hit frame
        ax2.axvline(x=gh_ground_truth, color='cyan', linestyle='-', alpha=0.5, label="Ground Hit (Ground Truth)")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("vy / ay")
    ax2.legend()
    ax2.set_title("Vertical Velocity (vy) and Acceleration (ay) over Frames")
    ax2.grid(True)

    
    plt.savefig(save_plot)
    plt.close()

def check_landing_inside_court(point, court_polygon):
    """
    Check if a point is inside a polygon using the ray casting algorithm.
    
    Args:
        point: Tuple (x, y) representing the point to check
        court_polygon: List of tuples/lists [(x, y), ...] representing polygon vertices in order
        
    Returns:
        bool: True if point is inside the polygon, False otherwise
    """
    if not court_polygon or len(court_polygon) < 3:
        return False
    
    # Convert point to scalars (handle numpy arrays, lists, tuples)
    x, y = float(point[0]), float(point[1])
    n = len(court_polygon)
    is_inside = False
    
    # Ray casting algorithm: cast a horizontal ray from point to infinity
    # Count how many edges it intersects
    p1 = court_polygon[0]
    # Convert to scalars, handling numpy arrays, lists, or tuples
    p1x, p1y = float(p1[0]), float(p1[1])
    
    for i in range(1, n + 1):
        p2 = court_polygon[i % n]
        # Convert to scalars, handling numpy arrays, lists, or tuples
        p2x, p2y = float(p2[0]), float(p2[1])
        
        # Check if the ray intersects with this edge
        # The ray is horizontal (y = constant), so we check if y is between the edge's y-values
        if min(p1y, p2y) < y <= max(p1y, p2y):
            # Edge is not horizontal, calculate x-intersection
            if p1y != p2y:
                # Calculate x-coordinate where the edge intersects the horizontal ray
                x_intersection = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                # If the point is to the left of the intersection, the ray crosses the edge
                if x <= x_intersection:
                    is_inside = not is_inside
            # Edge is horizontal (p1y == p2y), check if point is on the edge
            elif y == p1y and min(p1x, p2x) <= x <= max(p1x, p2x):
                # Point is on the boundary edge
                return True
        
        p1x, p1y = p2x, p2y
    
    return is_inside

def line_judge_decision(pred_dict, court_lines=None, save_plot=None, ground_hit_frame=None):

    # Compute features and ground hit candidate frames
    features = prepare_feature(pred_dict)
    gh_candidates = find_ground_hit_frame(features)
    gh_below_net_pole = find_ball_below_net_pole(gh_candidates, court_lines['n'])

    plot_ground_hit_data(features, gh_candidates, gh_below_net_pole, ground_hit_frame, save_plot)

    # Doubles court polygon
    # b0 = court_lines['b0']  # top boundary
    # b1 = court_lines['b1']  # bottom boundary
    
    # top_left = b0[0]
    # top_right= b0[1]
    # bottom_left = b1[0]
    # bottom_right = b1[1]

    # Singles court polygon
    s0 = court_lines['s0']  # top boundary
    s1 = court_lines['s1']  # bottom boundary
    
    top_left = s0[0]
    top_right= s1[0]
    bottom_left = s0[1]
    bottom_right = s1[1]
    
    # Define polygon vertices in counter-clockwise order
    court_polygon = [
        top_left,
        bottom_left,
        bottom_right,
        top_right,
    ]

    # For each ground hit detected below net pole, check IN/OUT using court polygon
    # Assume gh_below_net_pole is a DataFrame with columns "X", "Y"
    ground_hit_candidate_with_decision = gh_below_net_pole.copy()
    ground_hit_candidate_with_decision['decision'] = [
        "IN" if check_landing_inside_court((row['X'], row['Y']), court_polygon) else "OUT"
        for idx, row in gh_below_net_pole.iterrows()
    ]

    # Plot the court polygon and ground hit positions
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    court_poly_np = [court_polygon + [court_polygon[0]]]  # close the loop for the polygon
    court_xs = [pt[0] for pt in court_poly_np[0]]
    court_ys = [pt[1] for pt in court_poly_np[0]]

    plt.plot(court_xs, court_ys, 'b-', label='Court Polygon')

    # Plot ground hit positions with decision
    in_positions = ground_hit_candidate_with_decision[ground_hit_candidate_with_decision['decision'] == "IN"]
    out_positions = ground_hit_candidate_with_decision[ground_hit_candidate_with_decision['decision'] == "OUT"]

    plt.scatter(in_positions['X'], in_positions['Y'], c='g', label='IN', marker='o', s=60, zorder=3)
    plt.scatter(out_positions['X'], out_positions['Y'], c='r', label='OUT', marker='x', s=60, zorder=3)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.title("Court Polygon and Ground Hit Positions")

    if save_plot is not None:
        plt.savefig(save_plot.replace('.png','polygon.png'), bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    

    return ground_hit_candidate_with_decision



if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Ground-hit frame detection and line judge (IN/OUT) visualization")
    parser.add_argument("csv_file", help="Path to ball prediction result CSV (with columns Frame, X, Y, Visibility)")
    parser.add_argument("--court_lines", help="Path to pickle or JSON file of court lines", default=None)
    parser.add_argument("--save_plot", help="Path to save plot image instead of displaying", default=None)
    parser.add_argument("--ground_hit_frame", type=int, help="Frame number where ground hit occurs (optional)", default=None)
    args = parser.parse_args()

    # Load ball prediction result
    pred_dict = pd.read_csv(args.csv_file).to_dict(orient="list")

    # Load court lines
    if args.court_lines and args.court_lines.endswith(".pkl"):
        import pickle
        with open(args.court_lines, "rb") as f:
            court_lines = pickle.load(f)
    elif args.court_lines and args.court_lines.endswith(".json"):
        import json
        with open(args.court_lines, "r") as f:
            court_lines = json.load(f)


    if args.save_plot is None:
        save_plot = args.csv_file.split('/')[-1].split('.')[0].replace('.csv', '_plot.png')
    
    ground_hit_frame = None
    if args.ground_hit_frame is not None:
        ground_hit_frame = int(args.ground_hit_frame)

    # Run line_judge_decision (will also plot inside the function)
    print(save_plot)
    line_judge_decision(pred_dict, {}, save_plot, ground_hit_frame)
