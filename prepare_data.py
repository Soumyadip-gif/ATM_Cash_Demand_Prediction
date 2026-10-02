import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

print("\n========== DATA PREPARATION ==========\n")

# Load feature dataset
df = pd.read_csv("dataset/atm_features.csv")

# Convert Date
df["Date"] = pd.to_datetime(df["Date"])

# Sort chronologically
df = df.sort_values("Date").reset_index(drop=True)

# ==========================================
# Features and Target
# ==========================================

features = [
    "ATM_ID",
    "Weekday",
    "Month",
    "Previous_Day_Demand",
    "Previous_7_Day_Avg",
    "Working_day"
]

target = "Amount"

X = df[features]
y = df[target]

# ==========================================
# Time-Based Train/Test Split
# ==========================================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Total Records:", len(df))
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# ==========================================
# Categorical & Numerical Features
# ==========================================

categorical_features = [
    "ATM_ID",
    "Weekday",
    "Working_day"
]

numerical_features = [
    "Month",
    "Previous_Day_Demand",
    "Previous_7_Day_Avg"
]

# ==========================================
# Encoder
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

# Fit ONLY on training data
X_train_encoded = preprocessor.fit_transform(X_train)

# Transform test data
X_test_encoded = preprocessor.transform(X_test)

# ==========================================
# Save Prepared Data
# ==========================================

joblib.dump(
    preprocessor,
    "model/preprocessor.pkl"
)

joblib.dump(
    X_train_encoded,
    "model/X_train.pkl"
)

joblib.dump(
    X_test_encoded,
    "model/X_test.pkl"
)

joblib.dump(
    y_train,
    "model/y_train.pkl"
)

joblib.dump(
    y_test,
    "model/y_test.pkl"
)

print("\nEncoded Training Shape:", X_train_encoded.shape)
print("Encoded Testing Shape:", X_test_encoded.shape)

print("\nData preparation completed!")