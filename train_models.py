import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Create model folder
os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("adult.csv")

# Clean
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

# Convert categorical → numeric
df = pd.get_dummies(df)

# 🔥 SAVE ALL TRAINING COLUMNS
joblib.dump(df.columns, "model/columns.pkl")

# Target split
X = df.drop("income_<=50K", axis=1)
y = df["income_<=50K"]

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "model/scaler.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Models
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "dtree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "nb": GaussianNB(),
    "rf": RandomForestClassifier(),
    "xgb": XGBClassifier(eval_metric='logloss')
}

# Train + Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"model/{name}.pkl")

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]

    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("AUC:", roc_auc_score(y_test, probs))
    print("Precision:", precision_score(y_test, preds))
    print("Recall:", recall_score(y_test, preds))
    print("F1:", f1_score(y_test, preds))
    print("MCC:", matthews_corrcoef(y_test, preds))
