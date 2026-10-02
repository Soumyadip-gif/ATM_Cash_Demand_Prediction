from flask import Flask, render_template, request
import pandas as pd
import joblib


app = Flask(__name__)


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
# Home Page
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# Prediction
# ==============================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get inputs
        atm_id = request.form["atm_id"]
        weekday = request.form["weekday"]
        month = int(request.form["month"])

        previous_day = float(
            request.form["previous_day"]
        )

        previous_7_day_avg = float(
            request.form["previous_7_day_avg"]
        )

        working_day = request.form["working_day"]


        # Create input DataFrame
        input_data = pd.DataFrame({
            "ATM_ID": [atm_id],
            "Weekday": [weekday],
            "Month": [month],
            "Previous_Day_Demand": [previous_day],
            "Previous_7_Day_Avg": [previous_7_day_avg],
            "Working_day": [working_day]
        })


        # Preprocess
        input_encoded = preprocessor.transform(
            input_data
        )


        # Prediction
        prediction = model.predict(
            input_encoded
        )[0]


        prediction = max(
            0,
            prediction
        )


        # Format prediction
        formatted_prediction = (
            f"₹{prediction:,.2f}"
        )


        return render_template(
            "index.html",
            prediction=formatted_prediction
        )


    except Exception as e:

        return render_template(
            "index.html",
            error=str(e)
        )


# ==============================
# Run Application
# ==============================

if __name__ == "__main__":

    app.run(
        debug=True
    )