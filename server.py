from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Load and prepare your dataset
# Assuming phishing.csv is in C:\AWS\qr-code\phishing_detector\
try:
    data = pd.read_csv('C:/AWS/qr-code/phishing_detector/phishing.csv', index_col=None)
except FileNotFoundError:
    print("Error: phishing.csv not found at C:/AWS/qr-code/phishing_detector/phishing.csv")
    print("Please place phishing.csv in the specified directory or update the path below.")
    # If phishing.csv is in a different directory, update the path here, e.g.:
    # data = pd.read_csv('C:/path/to/your/phishing.csv', index_col=None)
    exit(1)

# Display the columns to help identify the target column
print("Columns in dataset:", data.columns.tolist())

# Replace 'target_column' with the actual name of your target column
# For example, if your target column is named 'class', use:
# X = data.drop('class', axis=1)
# y = data['class']
X = data.drop('target_column', axis=1)  # Update this
y = data['target_column']  # Update this

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction function for URLs
# Update this to match the features in phishing.csv
def extract_features_from_url(url):
    features = {
        'length': len(url),
        'has_digits': any(c.isdigit() for c in url),
        'num_slashes': url.count('/'),
        'has_https': int('https' in url),
        'num_dots': url.count('.')
    }
    return pd.DataFrame([features])

# Define the inference function
def MakeInference(predictors, url):
    X = extract_features_from_url(url)
    if isinstance(predictors, list):
        model = predictors[3]  # VotingClassifier at index 3
    else:
        model = predictors
    prediction = model.predict(X)
    probability = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    result = "Phishing" if prediction[0] == 1 else "Legitimate"
    return result, probability[0] if probability is not None else None

# Define and train predictors
predictors = [
    RandomForestClassifier(),
    DecisionTreeClassifier(),
    XGBClassifier(),
    VotingClassifier(estimators=[
        ('rf', RandomForestClassifier()),
        ('dt', DecisionTreeClassifier()),
        ('xgb', XGBClassifier())
    ], voting='soft')
]

# Train the models
print("Training models...")
for model in predictors:
    model.fit(X_train, y_train)
print("Models trained successfully!")

# Flask app
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        url = data.get('url', "https://google.com/4ssd")
    else:
        url = request.args.get('url', "https://google.com/4ssd")
    result, prob = MakeInference(predictors, url)
    # Display the prediction in the terminal for debugging
    print(f"URL: {url}")
    print(f"Prediction: {result}")
    print(f"Phishing Probability: {prob:.2f}" if prob is not None else "Probability not available")
    return jsonify({'prediction': result, 'probability': prob})

if __name__ == '__main__':
    app.run(host='localhost', port=5000)  # Running on localhost:5000