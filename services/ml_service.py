from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "category_model.joblib"
)

category_model = joblib.load(
    MODEL_FILE
)


def predict_category(description):
    predicted_category = category_model.predict(
        [description]
    )[0]

    probabilities = category_model.predict_proba(
        [description]
    )[0]

    confidence = max(probabilities)

    return (
        predicted_category,
        round(float(confidence), 3)
    )