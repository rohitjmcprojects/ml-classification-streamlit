import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Income Prediction Dashboard", layout="wide")

# ---------- HIDE STREAMLIT DEFAULT ICONS ----------
st.markdown("""
    <style>
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.title("Income Prediction using ML Models")
st.write("Upload dataset and predict income category using trained machine learning models.")

# ---------- SIDEBAR ----------
st.sidebar.header("Controls")
file = st.sidebar.file_uploader("Upload CSV")
model_name = st.sidebar.selectbox(
    "Select Model",
    ["logistic", "dtree", "knn", "nb", "rf", "xgb"]
)

# ---------- MAIN LOGIC ----------
if file:
    original_data = pd.read_csv(file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(original_data.head())

    # Keep original for display
    display_data = original_data.copy()

    # Convert categorical → numeric
    data = pd.get_dummies(original_data)

    # Load training columns
    train_cols = joblib.load("model/columns.pkl")
    train_cols = [col for col in train_cols if col != "income_<=50K"]

    # Match columns
    data = data.reindex(columns=train_cols, fill_value=0)

    # Load scaler & model
    scaler = joblib.load("model/scaler.pkl")
    model = joblib.load(f"model/{model_name}.pkl")

    X = scaler.transform(data)
    preds = model.predict(X)

    # Convert 0/1 → readable labels
    pred_labels = ["High Income (>50K)" if p == 1 else "Low Income (<=50K)" for p in preds]

    # ---------- SUMMARY METRICS ----------
    high_count = sum(preds)
    low_count = len(preds) - high_count

    st.subheader("Prediction Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(preds))
    col2.metric("High Income", high_count)
    col3.metric("Low Income", low_count)

    # ---------- RESULT TABLE ----------
    result_df = display_data.copy()

    # Add Sr No
    result_df.insert(0, "Sr No", range(1, len(result_df) + 1))

    # Add Prediction column
    result_df["Prediction"] = pred_labels

    # Show only important columns if they exist
    important_cols = ["Sr No"]

    for col in ["age", "education", "occupation", "hours.per.week"]:
        if col in result_df.columns:
            important_cols.append(col)

    important_cols.append("Prediction")

    result_df = result_df[important_cols]

    st.subheader("Prediction Results")
    st.dataframe(result_df)
