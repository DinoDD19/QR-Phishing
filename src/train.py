import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from src.preprocess import preprocess_data


FEATURE_NAMES = [
    "url_length",
    "hyphen_count",
    "dot_count",
    "slash_count",
    "domain_length",
    "https_presence"
]


def train_model(data_path, model_path, artifacts_path="models/model_artifacts.pkl"):
    """Train a RandomForest classifier on the prepared dataset and save artifacts."""
    
    # Load data
    data = pd.read_csv(data_path)
    
    # Preprocess with training flag
    data, artifacts = preprocess_data(data, training=True)
    
    # Extract features and target
    feature_cols = artifacts["feature_names"]
    X = data[feature_cols].values
    y = data["label"].values
    
    # Train-Test Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Successfully trained model!")
    
    # Evaluation
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print("\n--- Model Performance ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Phishing", "Safe"]))
    
    # Feature Importance
    feature_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)
    
    print("\n--- Feature Importance ---")
    print(feature_importance)
    
    # Save model
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save artifacts
    artifacts["model"] = model
    artifacts["feature_columns"] = feature_cols
    artifacts["threshold"] = 0.5
    artifacts["accuracy"] = accuracy
    artifacts["precision"] = precision
    artifacts["recall"] = recall
    artifacts["f1_score"] = f1
    artifacts["feature_importance"] = feature_importance
    
    os.makedirs(os.path.dirname(artifacts_path) or ".", exist_ok=True)
    joblib.dump(artifacts, artifacts_path)
    print(f"Artifacts saved to: {artifacts_path}")
