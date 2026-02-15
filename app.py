import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Bank Marketing Prediction", layout="wide")

st.title("Bank Marketing Prediction using ML Models")
st.write("Upload dataset OR click button to use sample test file.")

# ---------------- SIDEBAR ----------------
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

model_name = st.sidebar.selectbox(
    "Select Model",
    ["logistic", "dtree", "knn", "nb", "rf", "xgb"]
)

use_test_btn = st.sidebar.button("Use Sample Test Data")

# ---------------- LOAD TRAINED OBJECTS ----------------
scaler = joblib.load("model/scaler.pkl")
model = joblib.load(f"model/{model_name}.pkl")

# 🔴 CRITICAL: Get EXACT training feature order from scaler
train_cols = list(scaler.feature_names_in_)

# ---------------- FUNCTION ----------------
def run_prediction(df):

    df.columns = df.columns.str.strip()
    display_data = df.copy()

    # Remove target column if present
    if "y" in df.columns:
        df = df.drop("y", axis=1)

    # Replace unknown values
    df.replace("unknown", pd.NA, inplace=True)

    for col in df.select_dtypes(include="number").columns:
        df[col].fillna(df[col].median(), inplace=True)

    for col in df.select_dtypes(include="object").columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # One-hot encoding
    df = pd.get_dummies(df)

    # 🔴 FORCE EXACT FEATURE MATCH
    fixed_df = pd.DataFrame(columns=train_cols)

    for col in train_cols:
        if col in df.columns:
            fixed_df[col] = df[col]
        else:
            fixed_df[col] = 0

    # Scale
    X = scaler.transform(fixed_df)

    # Predict
    preds = model.predict(X)
    labels = ["Subscribed" if p == 1 else "Not Subscribed" for p in preds]

    # ---------------- SUMMARY ----------------
    st.subheader("Prediction Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", len(preds))
    c2.metric("Subscribed", int(sum(preds)))
    c3.metric("Not Subscribed", int(len(preds) - sum(preds)))

    # ---------------- TABLE ----------------
    result_df = display_data.copy()
    result_df.insert(0, "Sr No", range(1, len(result_df)+1))
    result_df["Prediction"] = labels

    st.subheader("Prediction Results")
    st.dataframe(result_df)


# ---------------- CASE 1: UPLOAD ----------------
if file:
    data = pd.read_csv(file, sep=None, engine="python")
    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())
    run_prediction(data)

# ---------------- CASE 2: AUTO TEST ----------------
elif use_test_btn:
    if os.path.exists("test.csv"):
        data = pd.read_csv("test.csv", sep=None, engine="python")
        st.success("Loaded test.csv automatically")
        st.dataframe(data.head())
        run_prediction(data)
    else:
        st.error("test.csv not found in project folder")
