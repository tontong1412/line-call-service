import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

test_pred_dict = {
    "Frame": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
    ],
    "X": [
        667,
        660,
        650,
        640,
        630,
        622,
        612,
        602,
        595,
        585,
        575,
        565,
        555,
        545,
        535,
        525,
        515,
        505,
        495,
        482,
        470,
        460,
        447,
        435,
        422,
        412,
        397,
        385,
        372,
        360,
        350,
        332,
        320,
        305,
        295,
        280,
        265,
        252,
        240,
        227,
        215,
        207,
        207,
        205,
        205,
        205,
        202,
        209,
        211,
        235,
        268,
        317,
        328,
        448,
        516,
        525,
        436,
        414,
        336,
        310,
        266,
        245,
        267,
        247,
        215,
        215,
        217,
        215,
        215,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    "Y": [
        405,
        392,
        380,
        367,
        354,
        347,
        336,
        330,
        320,
        315,
        307,
        302,
        296,
        295,
        290,
        290,
        287,
        287,
        290,
        290,
        295,
        297,
        302,
        310,
        317,
        327,
        337,
        350,
        362,
        375,
        387,
        410,
        425,
        445,
        460,
        485,
        510,
        535,
        560,
        587,
        615,
        630,
        615,
        607,
        602,
        600,
        602,
        595,
        587,
        579,
        546,
        497,
        468,
        444,
        400,
        406,
        352,
        406,
        483,
        515,
        553,
        596,
        579,
        608,
        640,
        640,
        640,
        642,
        642,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    "Visibility": [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
}

test_court_coord = {
    "center_line_top": [
        [703.9299926757812, 479.5299987792969],
        [707.6199951171875, 499.8399963378906],
    ],
    "center_line_bottom": [
        [739.6400146484375, 676.0900268554688],
        [713.9400024414062, 534.6400146484375],
    ],
    "long_service_line_bottom": [
        [300.5400085449219, 611.3099975585938],
        [1112.4599609375, 651.469970703125],
    ],
    "long_service_line_top": [
        [593.469970703125, 473.9800109863281],
        [811.6500244140625, 489.760009765625],
    ],
    "outer_boundary_bottom": [
        [210.69000244140625, 653.4400024414062],
        [1194.5, 695.5700073242188],
    ],
    "outer_boundary_left": [
        [598.0499877929688, 471.8299865722656],
        [210.69000244140625, 653.4400024414062],
    ],
    "outer_boundary_right": [
        [806.489990234375, 486.989990234375],
        [1194.5, 695.5700073242188],
    ],
    "outer_boundary_top": [
        [598.0499877929688, 471.8299865722656],
        [806.489990234375, 486.989990234375],
    ],
    "short_service_line_bottom": [
        [494.239990234375, 520.5],
        [919.77001953125, 547.8900146484375],
    ],
    "short_service_line_top": [
        [560.219970703125, 489.57000732421875],
        [848.6599731445312, 509.6600036621094],
    ],
    "singles_left_line": [
        [614.239990234375, 473.010009765625],
        [295.9100036621094, 657.0900268554688],
    ],
    "singles_right_line": [
        [791.22998046875, 485.8800048828125],
        [1130.02001953125, 692.8099975585938],
    ],
}


def line_judge_decision(pred_dict, court_coord):
    frame_at_ground_hit = find_ground_hit_frame(pred_dict)
    frame_at_ground_hit_with_decision = pd.DataFrame(frame_at_ground_hit)

    frame_at_ground_hit_with_decision["decision"] = [
        is_point_outside_court_precise((x, y), court_coord)
        for x, y in zip(
            frame_at_ground_hit_with_decision["X"],
            frame_at_ground_hit_with_decision["Y"],
        )
    ]

    frame_at_ground_hit_with_decision["decision"] = frame_at_ground_hit_with_decision[
        "decision"
    ].map({True: "OUT", False: "IN"})
    print("------")
    print(frame_at_ground_hit_with_decision)
    return frame_at_ground_hit_with_decision


def find_ground_hit_frame(pred_dict):
    features, num_frames = prepare_features(pred_dict)
    # print(features)
    is_bounce_frames = features.loc[
        (abs(features["y_div_1"]) > 1e10) | (abs(features["y_div_2"]) > 1e10)
    ]
    print(is_bounce_frames)
    return is_bounce_frames


def is_point_outside_court_precise(point, court_lines):
    """
    Checks if a point is outside the non-rectangular court.
    """
    # Extract the four corner points of the court polygon
    polygon_vertices = [
        court_lines["outer_boundary_left"][0],  # Top-left
        court_lines["outer_boundary_right"][0],  # Top-right
        court_lines["outer_boundary_bottom"][1],  # Bottom-right
        court_lines["outer_boundary_bottom"][0],  # Bottom-left
    ]

    # visualize(point, polygon_vertices)

    # The Ray Casting algorithm returns true if inside, so we negate the result
    return not is_inside_polygon(point, polygon_vertices)


def is_inside_polygon(point, polygon):
    """
    Determines if a point is inside a polygon using the Ray Casting Algorithm.
    The polygon is a list of (x, y) vertices.
    """
    x, y = point
    n = len(polygon)
    is_inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_intersection = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= x_intersection:
                        is_inside = not is_inside
        p1x, p1y = p2x, p2y

    return is_inside


def prepare_features(pred_dict):

    # read ball prediction result
    x_pred, y_pred, vis_pred, frame_i = (
        pred_dict["X"],
        pred_dict["Y"],
        pred_dict["Visibility"],
        pred_dict["Frame"],
    )

    labels = pd.DataFrame(
        {
            "frame": pred_dict["Frame"],
            "x-coordinate": pred_dict["X"],
            "y-coordinate": pred_dict["Y"],
        }
    )

    num = 3
    eps = 1e-15
    for i in range(1, num):
        labels["Frame"] = labels["frame"]
        labels["X"] = labels["x-coordinate"]
        labels["Y"] = labels["y-coordinate"]
        labels["x_lag_{}".format(i)] = labels["x-coordinate"].shift(i)
        labels["x_lag_inv_{}".format(i)] = labels["x-coordinate"].shift(-i)
        labels["y_lag_{}".format(i)] = labels["y-coordinate"].shift(i)
        labels["y_lag_inv_{}".format(i)] = labels["y-coordinate"].shift(-i)
        labels["x_diff_{}".format(i)] = abs(
            labels["x_lag_{}".format(i)] - labels["x-coordinate"]
        )
        labels["y_diff_{}".format(i)] = (
            labels["y_lag_{}".format(i)] - labels["y-coordinate"]
        )
        labels["x_diff_inv_{}".format(i)] = abs(
            labels["x_lag_inv_{}".format(i)] - labels["x-coordinate"]
        )
        labels["y_diff_inv_{}".format(i)] = (
            labels["y_lag_inv_{}".format(i)] - labels["y-coordinate"]
        )
        labels["x_div_{}".format(i)] = abs(
            labels["x_diff_{}".format(i)] / (labels["x_diff_inv_{}".format(i)] + eps)
        )
        labels["y_div_{}".format(i)] = labels["y_diff_{}".format(i)] / (
            labels["y_diff_inv_{}".format(i)] + eps
        )

    for i in range(1, num):
        labels = labels[labels["x_lag_{}".format(i)].notna()]
        labels = labels[labels["x_lag_inv_{}".format(i)].notna()]
    labels = labels[labels["x-coordinate"].notna()]

    colnames_x = (
        ["x_diff_{}".format(i) for i in range(1, num)]
        + ["x_diff_inv_{}".format(i) for i in range(1, num)]
        + ["x_div_{}".format(i) for i in range(1, num)]
        + ["X"]
    )
    colnames_y = (
        ["y_diff_{}".format(i) for i in range(1, num)]
        + ["y_diff_inv_{}".format(i) for i in range(1, num)]
        + ["y_div_{}".format(i) for i in range(1, num)]
        + ["Y"]
    )

    colnames_general = ["Frame"]
    colnames = colnames_general + colnames_x + colnames_y

    features = labels[colnames]
    return features, list(labels["frame"])


def visualize(point, court):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a polygon patch
    polygon_patch = Polygon(
        court,
        closed=True,
        edgecolor="blue",
        facecolor="lightblue",
        linewidth=2,
        alpha=0.5,
    )
    ax.add_patch(polygon_patch)

    print(point)

    # Plot the test point
    ax.scatter(point[0], point[1], color="red", s=100, zorder=5)
    ax.text(point[0] + 50, point[1], "Test Point", color="red", fontsize=12)

    # Set the axis limits to ensure everything is visible
    min_x = min(v[0] for v in court)
    max_x = max(v[0] for v in court)
    min_y = min(v[1] for v in court)
    max_y = max(v[1] for v in court)

    # ax.set_xlim(min_x - 100, max(max_x + 100, point[0] + 150))
    # ax.set_ylim(min_y - 50, max_y + 50)

    ax.invert_yaxis()

    ax.set_title("Non-Rectangular Polygon with Test Point")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")

    # Display the plot
    plt.show()


# line_judge_decision(test_pred_dict, test_court_coord)
