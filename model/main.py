import os
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "features.pkl")


def load_data():
    data = pd.read_csv(DATA_PATH)

    # Drop unnecessary columns
    data = data.drop(columns=['Unnamed: 32', 'id'], errors='ignore')

    # Encode target
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data


def train_model(data):
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    # Save feature order
    feature_names = X.columns.tolist()

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train
    model = LogisticRegression(max_iter=10000, class_weight='balanced')
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model, scaler, feature_names


def save_artifacts(model, scaler, feature_names):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(feature_names, f)


def main():
    data = load_data()
    model, scaler, feature_names = train_model(data)
    save_artifacts(model, scaler, feature_names)


if __name__ == "__main__":
    main()
