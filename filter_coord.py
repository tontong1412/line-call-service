import pandas as pd

df = pd.read_csv("prediction/11_09_2025_22:18:23_ball.csv")

coords_to_remove = [
    (902, 305),
    (910, 291),
    (912, 282),
    (914, 268)
]

# Remove rows that match any of those (X, Y) pairs
df_filtered = df[~df[['X', 'Y']].apply(tuple, axis=1).isin(coords_to_remove)]

# Save the cleaned data back to CSV (optional)
df_filtered.to_csv("cleaned_file.csv", index=False)