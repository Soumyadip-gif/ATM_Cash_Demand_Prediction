import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


print("\n========== IMPROVED MODEL TRAINING ==========\n")


# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("dataset/atm_features.csv")

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)


# ==============================
# Six Input Features
# ==============================

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


# ==============================
# Time-Based Split
# ==============================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print("Training Records:", len(X_train))
print("Testing Records :", len(X_test))


# ==============================
# Preprocessing
# ==============================

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


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
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


print("\nEncoding data...")

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)


print("Encoded Training Shape:", X_train_encoded.shape)
print("Encoded Testing Shape :", X_test_encoded.shape)


# ==============================
# XGBoost Model
# ==============================

model = XGBRegressor(
    n_estimators=150,
    max_depth=8,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=42
)


print("\nTraining XGBoost...")
print("Please wait...\n")


model.fit(
    X_train_encoded,
    y_train
)


# ==============================
# Prediction
# ==============================

print("Generating predictions...")

predictions = model.predict(X_test_encoded)


# ==============================
# Evaluation
# ==============================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n========== IMPROVED MODEL RESULTS ==========\n")

print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R²   : {r2:.4f}")

print("\n============================================")


# ==============================
# Save Model
# ==============================

joblib.dump(
    model,
    "model/atm_demand_model.pkl"
)

joblib.dump(
    preprocessor,
    "model/preprocessor.pkl"
)


print("\nImproved model saved successfully!")

print("Model       : model/atm_demand_model.pkl")
print("Preprocessor: model/preprocessor.pkl")

print("\n========== TRAINING COMPLETED ==========\n")