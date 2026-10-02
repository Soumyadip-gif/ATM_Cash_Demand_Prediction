import pandas as pd
import joblib
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("\n========== MODEL TRAINING ==========\n")


# ==========================================
# Load Prepared Data
# ==========================================

X_train = joblib.load("model/X_train.pkl")
X_test = joblib.load("model/X_test.pkl")

y_train = joblib.load("model/y_train.pkl")
y_test = joblib.load("model/y_test.pkl")

print("Original Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# ==========================================
# Use Recent Training Data
# ==========================================

TRAIN_SIZE = 100000

if len(X_train) > TRAIN_SIZE:

    X_train_model = X_train[-TRAIN_SIZE:]
    y_train_model = y_train.iloc[-TRAIN_SIZE:]

else:

    X_train_model = X_train
    y_train_model = y_train


print(
    "Training Data Used:",
    X_train_model.shape
)


# ==========================================
# Model
# ==========================================

model = HistGradientBoostingRegressor(
    max_iter=100,
    learning_rate=0.10,
    max_leaf_nodes=31,
    random_state=42
)


# ==========================================
# Train
# ==========================================

print("\nTraining HistGradientBoosting...")

model.fit(
    X_train_model,
    y_train_model
)


# ==========================================
# Prediction
# ==========================================

print("\nGenerating predictions...")

predictions = model.predict(X_test)


# ==========================================
# Evaluation
# ==========================================

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


print("\n========== MODEL RESULTS ==========\n")

print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R²   : {r2:.4f}")


# ==========================================
# Save Model
# ==========================================

joblib.dump(
    model,
    "model/atm_demand_model.pkl"
)

print(
    "\nModel saved successfully:"
    "\nmodel/atm_demand_model.pkl"
)

print("\n========== TRAINING COMPLETED ==========\n")