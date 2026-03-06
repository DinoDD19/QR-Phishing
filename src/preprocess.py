import pandas as pd
import random
import joblib
import os
from src.features import extract_features


def create_dataset(input_file, output_file):
    """Create a labeled dataset from safe URLs and generate phishing variants."""
    df = pd.read_excel(input_file)

    safe_urls = df.iloc[:, 0].dropna().unique().tolist()

    fake_urls = [
        f"https://{random.randint(1000, 9999)}-{url}"
        for url in safe_urls
    ]

    data = pd.DataFrame({
        "url": safe_urls + fake_urls,
        "label": [1] * len(safe_urls) + [0] * len(fake_urls)
    })

    data["features"] = data["url"].apply(extract_features)

    data.to_csv(output_file, index=False)

    print("Dataset saved to:", output_file)


def preprocess_data(df, training=True, artifacts=None):
    """
    Preprocess URL data for training or inference.
    
    If training=True:
        - Return transformed df + artifacts
    If training=False:
        - Use passed artifacts
        - Return transformed df only
    """
    
    if training:
        artifacts = {}
        feature_names = [
            "url_length",
            "hyphen_count",
            "dot_count", 
            "slash_count",
            "domain_length",
            "https_presence"
        ]
        artifacts["feature_names"] = feature_names
    else:
        feature_names = artifacts["feature_names"]
    
    # Extract features from URL column
    df["features"] = df["url"].apply(extract_features)
    
    # Convert features list to separate columns
    features_array = df["features"].apply(pd.Series)
    features_array.columns = feature_names
    
    # Combine with original dataframe
    df = pd.concat([df[["url", "label"]], features_array], axis=1)
    
    return (df, artifacts) if training else df
