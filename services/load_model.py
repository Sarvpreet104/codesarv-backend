import joblib

MODELS = {
    "binary-logistic-regression": {
        "model": joblib.load("models/logistic_regression.pkl"),
        "metrics_path": "data/binary-logistic-regression/metrics.json",
        "test_data_path": "data/binary-logistic-regression/data.csv",
    },  
}

if __name__ == '__main__':
    print(MODELS.get("binary-logistic-regression").get("metrics_path"))