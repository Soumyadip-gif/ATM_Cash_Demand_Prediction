# ATM Cash Demand Prediction

A Machine Learning web application that predicts the expected cash demand of an ATM using historical transaction and demand patterns.

## Project Overview

This project uses **XGBoost Regression** to estimate the required ATM cash amount based on six important inputs:

- ATM ID
- Weekday
- Month
- Previous Day Cash Demand
- Previous 7-Day Average Demand
- Working Day / Holiday

The trained ML model is integrated with a **Flask web application** for real-time predictions.

## Features

- Machine Learning based cash demand prediction
- ATM-specific demand patterns
- Historical demand features
- XGBoost Regression model
- Real-time prediction through Flask
- Professional responsive web interface
- Animated prediction interface
- Error handling and input validation
- Ready for deployment with Gunicorn

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Flask
- HTML
- CSS
- JavaScript
- Gunicorn

## Machine Learning

**Algorithm:** XGBoost Regressor

**Target:** ATM Cash Demand (`Amount`)

### Model Performance

- **MAE:** ₹91,512.21
- **RMSE:** ₹158,902.78
- **R² Score:** 0.4703

The model provides a baseline prediction of ATM cash requirements and can be further improved with additional historical and operational features.

## Project Structure

```text
ATM_Cash_Demand_Prediction/
│
├── dataset/
├── model/
├── static/
│   ├── css/
│   └── js/
├── templates/
├── app.py
├── train.py
├── train_improved.py
├── clean_data.py
├── feature_engineering.py
├── prepare_data.py
├── evaluate_model.py
├── error_analysis.py
├── requirements.txt
├── Procfile
└── README.md
