import streamlit as st
import pandas as pd
import joblib

st.title("ML Classification Dashboard")

file = st.file_uploader("Upload Test CSV")

model_name = st.selectbox(
    "Select Model",
    ["logistic","dtree","knn","nb","rf","xgb"]
)

if file:
    data = pd.read_csv(file)

    # Convert categorical to numeric
    data = pd.get_dummies(data)

    # Load training columns
    train_cols = joblib.load("model/columns.pkl")

    # REMOVE TARGET COLUMN FROM TRAINING COLUMNS
    train_cols = [col for col in train_cols if col != "income_<=50K"]

    # Match columns
    data = data.reindex(columns=train_cols, fill_value=0)

    # Load scaler & model
    scaler = joblib.load("model/scaler.pkl")
    model = joblib.load(f"model/{model_name}.pkl")

    X = scaler.transform(data)

    preds = model.predict(X)

    st.subheader("Predictions")
    st.write(preds)
