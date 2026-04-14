from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from fastapi import FastAPI
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Load model once (IMPORTANT)
model = joblib.load("breast_cancer_cls_v1.pkl")

# Load dataset
df_test = pd.read_csv("breast-cancer-test.csv")
# Load evaluation data
y_test = pd.read_csv("breast-cancer-y-test.csv").squeeze()
y_pred = pd.read_csv("breast-cancer-y-pred.csv").squeeze()
y_prob = pd.read_csv("breast-cancer-y-prob.csv").squeeze()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "API running"}

@app.get("/predict")
def predict():
    # randomly sample 1 row
    sample = df_test.sample(n=1)

    prediction = model.predict(sample)

    return {
        "input": sample.to_dict(orient="records")[0],
        "prediction": prediction.tolist()
    }

@app.get("/performance")
def performance():
    try:
        # Core metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # ROC-AUC (requires probabilities)
        roc_auc = roc_auc_score(y_test, y_prob)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred).tolist()

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": cm
        }

    except Exception as e:
        return {"error": str(e)}