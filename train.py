import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import joblib

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    args = parser.parse_args()

    # Load data
    data_path = os.path.join(args.data_dir, "phishing.csv")
    data = pd.read_csv(data_path)
    X = data.drop('class', axis=1)
    y = data['class'].apply(lambda x: 1 if x == 1 else 0)  # Convert -1/1 to 0/1

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define classifiers
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    dt_clf = DecisionTreeClassifier(random_state=42)
    xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    voting_clf = VotingClassifier(estimators=[
        ('rf', rf_clf), ('dt', dt_clf), ('xgb', xgb_clf)], voting='soft')

    # Train
    voting_clf.fit(X_train, y_train)

    # Save model
    joblib.dump(voting_clf, os.path.join(args.model_dir, "model.pkl"))

    # Evaluate (optional logging)
    train_acc = voting_clf.score(X_train, y_train)
    test_acc = voting_clf.score(X_test, y_test)
    print(f"Training Accuracy: {train_acc}")
    print(f"Testing Accuracy: {test_acc}")