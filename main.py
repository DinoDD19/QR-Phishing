from src.preprocess import create_dataset
from src.train import train_model
from src.predict import predict_url

DATASET_INPUT = "data/dataseturl.xlsx"
DATASET_OUTPUT = "data/labeled_urls.csv"
MODEL_PATH = "models/url_classifier.pkl"
ARTIFACTS_PATH = "artifacts/model_artifacts.pkl"


def main():
    """Main orchestration function for the phishing detection pipeline."""
    
    print("\n" + "="*60)
    print("URL PHISHING DETECTION PIPELINE")
    print("="*60)
    
    print("\n[1/3] Creating dataset...")
    create_dataset(DATASET_INPUT, DATASET_OUTPUT)

    print("\n[2/3] Training model...")
    train_model(DATASET_OUTPUT, MODEL_PATH, ARTIFACTS_PATH)

    print("\n[3/3] Testing predictions...")
    test_urls = [
        "https://paypal-login-security.com",
        "https://www.paypal.com",
        "https://secure-amazon-account.net"
    ]
    
    print("\n" + "-"*60)
    for url in test_urls:
        result = predict_url(url, ARTIFACTS_PATH)
        print(f"\nURL: {result['url']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Phishing Probability: {result['probability_phishing']:.4f}")
        print(f"Safe Probability: {result['probability_safe']:.4f}")
        print(f"Confidence: {result['confidence']:.4f}")
    print("-"*60)


if __name__ == "__main__":
    main()
