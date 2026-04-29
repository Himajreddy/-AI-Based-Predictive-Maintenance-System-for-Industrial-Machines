import pandas as pd
from pathlib import Path

# Column names for NASA CMAPSS FD001
cols = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Load raw NASA file
df = pd.read_csv(
    "data/raw/train_FD001.txt",
    sep=r"\s+",
    header=None
)

# Keep only the first 26 valid columns
df = df.iloc[:, :26]
df.columns = cols

# Compute Remaining Useful Life (RUL)
max_cycles = df.groupby("unit")["cycle"].transform("max")
df["rul"] = max_cycles - df["cycle"]

# Define failure label:
# failure = 1 if RUL <= 20, else 0
df["failure"] = (df["rul"] <= 20).astype(int)

# Map CMAPSS sensors into your project schema
final_df = pd.DataFrame({
    "temperature": df["sensor_11"],
    "vibration": df["sensor_12"],
    "pressure": df["sensor_7"],
    "voltage": df["sensor_9"],
    "runtime_hours": df["cycle"],
    "failure": df["failure"],
    "rul": df["rul"]
})

# Save final dataset
Path("data/raw").mkdir(parents=True, exist_ok=True)
final_df.to_csv("data/raw/sensor_data.csv", index=False)

print("Saved successfully: data/raw/sensor_data.csv")