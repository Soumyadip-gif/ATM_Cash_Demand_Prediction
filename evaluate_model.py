import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("\n========== MODEL VALIDATION ==========\n")


# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("dataset/atm_features.csv")

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)


# ==============================
# Six Inputs
# ==============================

features = [
    "ATM_ID",
    "Weekday",
    "Month",
    "Previous_Day_Demand",
    "Previous_7_Day_Avg",
    "Working_day"
]

X = df[features]
y = df["Amount"]


# ==============================
# Same Test Split
# ==============================

split_index = int(len(df) * 0.80)

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]


# ==============================
# Load Model
# ==============================

model = joblib.load(
    "model/atm_demand_model.pkl"
)

preprocessor = joblib.load(
    "model/preprocessor.pkl"
)


# ==============================
# Transform Test Data
# ==============================

X_test_encoded = preprocessor.transform(X_test)


# ==============================
# Predictions
# ==============================

predictions = model.predict(
    X_test_encoded
)


# ==============================
# Metrics
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


print("MAE :", f"{mae:,.2f}")
print("RMSE:", f"{rmse:,.2f}")
print("R²  :", f"{r2:.4f}")


# ==============================
# Sample Predictions
# ==============================

results = pd.DataFrame({
    "Actual_Amount": y_test.values,
    "Predicted_Amount": predictions
})

results["Difference"] = (
    results["Actual_Amount"]
    - results["Predicted_Amount"]
)


print("\n========== SAMPLE PREDICTIONS ==========\n")

print(
    results.head(10).to_string(
        index=False
    )
)


# ==============================
# Save Results
# ==============================

results.to_csv(
    "dataset/prediction_results.csv",
    index=False
)

print(
    "\nPrediction results saved to:"
    "\ndataset/prediction_results.csv"
)

print("\n========== VALIDATION COMPLETED ==========\n")