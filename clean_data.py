import pandas as pd

print("\n========== DATA CLEANING ==========\n")

# Load dataset
df = pd.read_csv("dataset/atms_data.csv")

print("Original Shape:", df.shape)

# Remove unnecessary index column
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# Convert Date
df["Date"] = pd.to_datetime(
    df["Date"].astype(str),
    format="%Y%m%d"
)

# Check duplicates
duplicates = df.duplicated().sum()
print("Duplicate Rows:", duplicates)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Keep required columns
df = df[
    [
        "Date",
        "ATM_ID",
        "Number_of_Trxs",
        "Amount",
        "Weekday",
        "Working_day"
    ]
]

# Sort by ATM and Date
df.sort_values(
    by=["ATM_ID", "Date"],
    inplace=True
)

# Reset index
df.reset_index(drop=True, inplace=True)

print("\nCleaned Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# Save cleaned dataset
df.to_csv(
    "dataset/atm_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")
