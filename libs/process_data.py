import pandas as pd

def mark_default_shuttlecock_position(pred_dict):

    # TrackNetV3 sometimes return weird shuttle position, possibly default position
    # when it cannot detect the shuttlecock but still set visible to True.
    # This function will mark these coordinates to Visible to False so it is not
    # considered in ground hit detection.

    coords_from_default_value = [
        (902, 305),
        (910, 291),
        (912, 282),
        (914, 268), # HD video
        (609, 178), # 720p video
    ]

    # Set visibility to 0 for points at default positions
    xs = pred_dict["X"]
    ys = pred_dict["Y"]
    vis = pred_dict["Visibility"]
    for idx, (x, y) in enumerate(zip(xs, ys)):
        if (x, y) in coords_from_default_value:
            pred_dict["Visibility"][idx] = 0

    return pred_dict