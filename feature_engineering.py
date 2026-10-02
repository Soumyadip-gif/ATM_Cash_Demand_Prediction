import pandas as pd
import numpy as np

print("\n========== FEATURE ENGINEERING ==========\n")

# Load cleaned dataset
df = pd.read_csv("dataset/atm_cleaned.csv")

# Convert Date
df["Date"] = pd.to_datetime(df["Date"])

# Sort properly
df = df.sort_values(["ATM_ID", "Date"]).reset_index(drop=True)

# ==================================================
# 1. Month
# ==================================================

df["Month"] = df["Date"].dt.month


# ==================================================
# 2. Previous Day Cash Demand
# ==================================================

# Shift within each ATM
df["Previous_Day_Demand"] = (
    df.groupby("ATM_ID")["Amount"].shift(1)
)


# ==================================================
# 3. Previous 7-Day Average Demand
# ==================================================

df["Previous_7_Day_Avg"] = (
    df.groupby("ATM_ID")["Amount"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=7,
            min_periods=1
        ).mean()
    )
)


# ==================================================
# 4. Select Required Columns
# ==================================================

df = df[
    [
        "Date",
        "ATM_ID",
        "Weekday",
        "Month",
        "Previous_Day_Demand",
        "Previous_7_Day_Avg",
        "Working_day",
        "Amount"
    ]
]


# ==================================================
# 5. Remove rows without historical demand
# ==================================================

df.dropna(
    subset=[
        "Previous_Day_Demand",
        "Previous_7_Day_Avg"
    ],
    inplace=True
)


# ==================================================
# 6. Reset Index
# ==================================================

df.reset_index(drop=True, inplace=True)


# ==================================================
# 7. Save Feature Dataset
# ==================================================

df.to_csv(
    "dataset/atm_features.csv",
    index=False
)


# ==================================================
# 8. Display Results
# ==================================================

print("Feature Dataset Shape:", df.shape)

print("\nFeatures:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFeature engineering completed!")