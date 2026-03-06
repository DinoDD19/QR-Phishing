import joblib
import numpy as np
import pandas as pd
import os

from src.features import extract_features
from src.preprocess import preprocess_data


def predict_url(url, artifacts_path="models/model_artifacts.pkl"):
    """
    Predict whether a URL is safe or phishing.
    
    Returns:
        dict: {
            "url": input URL,
            "prediction": "Safe URL" or "Phishing URL",
            "probability_phishing": float,
            "probability_safe": float,
            "confidence": float
        }
    """
    
    # Load artifacts
    artifacts = joblib.load(artifacts_path)
    model = artifacts["model"]
    feature_columns = artifacts["feature_columns"]
    threshold = artifacts.get("threshold", 0.5)
    
    # Create DataFrame with URL
    df = pd.DataFrame({"url": [url], "label": [1]})  # dummy label
    
    # Preprocess with training=False
    df = preprocess_data(df, training=False, artifacts=artifacts)
    
    # Ensure correct feature columns order
    df = df[feature_columns]
    
    # Get prediction and probability
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    
    prob_phishing = probabilities[0]  # Class 0 = Phishing
    prob_safe = probabilities[1]      # Class 1 = Safe
    
    # Apply threshold
    final_prediction = 1 if prob_safe >= threshold else 0
    
    if final_prediction == 1:
        result = "Safe URL"
    else:
        result = "Phishing URL"
    
    confidence = max(prob_phishing, prob_safe)
    
    return {
        "url": url,
        "prediction": result,
        "probability_phishing": float(prob_phishing),
        "probability_safe": float(prob_safe),
        "confidence": float(confidence)
    }
