import warnings

import joblib
import pandas as pd
import sklearn
from datetime import datetime

from schemas import HousePredictionRequest, PredictionResponse

# Load model and preprocessor
MODEL_PATH = "models/trained/house_price_model.pkl"
PREPROCESSOR_PATH = "models/trained/preprocessor.pkl"


def _load_sklearn_artifact(path: str):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        artifact = joblib.load(path)
    for warning in caught:
        message = str(warning.message)
        if "Trying to unpickle estimator" in message and "when using version" in message:
            raise RuntimeError(
                f"Incompatible scikit-learn version for {path}: {message} "
                f"Re-create the artifact with scikit-learn {sklearn.__version__} "
                "and rebuild the Docker image."
            )
    return artifact


try:
    model = _load_sklearn_artifact(MODEL_PATH)
    preprocessor = _load_sklearn_artifact(PREPROCESSOR_PATH)
    if not hasattr(preprocessor, "_name_to_fitted_passthrough"):
        raise RuntimeError(
            "Preprocessor is incompatible with this scikit-learn version. "
            f"Re-run feature engineering with scikit-learn {sklearn.__version__} "
            "and rebuild the Docker image."
        )
except Exception as e:
    raise RuntimeError(f"Error loading model or preprocessor: {str(e)}")

def predict_price(request: HousePredictionRequest) -> PredictionResponse:
    """
    Predict house price based on input features.
    """
    # Prepare input data
    input_data = pd.DataFrame([request.dict()])
    input_data['house_age'] = datetime.now().year - input_data['year_built']
    input_data['bed_bath_ratio'] = input_data['bedrooms'] / input_data['bathrooms']
    input_data['price_per_sqft'] = 0  # Dummy value for compatibility

    # Preprocess input data
    processed_features = preprocessor.transform(input_data)

    # Make prediction
    predicted_price = model.predict(processed_features)[0]

    # Convert numpy.float32 to Python float and round to 2 decimal places
    predicted_price = round(float(predicted_price), 2)

    # Confidence interval (10% range)
    confidence_interval = [predicted_price * 0.9, predicted_price * 1.1]

    # Convert confidence interval values to Python float and round to 2 decimal places
    confidence_interval = [round(float(value), 2) for value in confidence_interval]

    return PredictionResponse(
        predicted_price=predicted_price,
        confidence_interval=confidence_interval,
        features_importance={},
        prediction_time=datetime.now().isoformat()
    )

def batch_predict(requests: list[HousePredictionRequest]) -> list[float]:
    """
    Perform batch predictions.
    """
    input_data = pd.DataFrame([req.dict() for req in requests])
    input_data['house_age'] = datetime.now().year - input_data['year_built']
    input_data['bed_bath_ratio'] = input_data['bedrooms'] / input_data['bathrooms']
    input_data['price_per_sqft'] = 0  # Dummy value for compatibility

    # Preprocess input data
    processed_features = preprocessor.transform(input_data)

    # Make predictions
    predictions = model.predict(processed_features)
    return predictions.tolist()