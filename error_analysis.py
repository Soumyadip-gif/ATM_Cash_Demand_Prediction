import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print("\n========== ERROR ANALYSIS ==========\n")


# ==============================
# Load Prediction Results
# ==============================

df = pd.read_csv(
    "dataset/prediction_results.csv"
)


# ==============================
# Calculate Errors
# ==============================

df["Absolute_Error"] = (
    abs(df["Difference"])
)

df["Percentage_Error"] = (
    df["Absolute_Error"]
    / df["Actual_Amount"].replace(0, np.nan)
) * 100


# ==============================
# Statistics
# ==============================

print("Average Absolute Error:")
print(
    f"₹{df['Absolute_Error'].mean():,.2f}"
)

print("\nMedian Absolute Error:")
print(
    f"₹{df['Absolute_Error'].median():,.2f}"
)

print("\nMaximum Absolute Error:")
print(
    f"₹{df['Absolute_Error'].max():,.2f}"
)


print("\nAverage Percentage Error:")
print(
    f"{df['Percentage_Error'].mean():.2f}%"
)


# ==============================
# Best Predictions
# ==============================

print("\n========== CLOSEST PREDICTIONS ==========\n")

best = df.sort_values(
    "Absolute_Error"
).head(10)

print(
    best[
        [
            "Actual_Amount",
            "Predicted_Amount",
            "Difference"
        ]
    ].to_string(index=False)
)


# ==============================
# Largest Errors
# ==============================

print("\n========== LARGEST ERRORS ==========\n")

worst = df.sort_values(
    "Absolute_Error",
    ascending=False
).head(10)

print(
    worst[
        [
            "Actual_Amount",
            "Predicted_Amount",
            "Difference"
        ]
    ].to_string(index=False)
)


# ==============================
# Actual vs Predicted Plot
# ==============================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Actual_Amount"],
    df["Predicted_Amount"],
    alpha=0.3
)

plt.xlabel("Actual Cash Demand")
plt.ylabel("Predicted Cash Demand")
plt.title("Actual vs Predicted ATM Cash Demand")

plt.tight_layout()

plt.savefig(
    "dataset/actual_vs_predicted.png"
)

plt.show()


print(
    "\nGraph saved to:"
    "\ndataset/actual_vs_predicted.png"
)

print("\n========== ANALYSIS COMPLETED ==========\n")