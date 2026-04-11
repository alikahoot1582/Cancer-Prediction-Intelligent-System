import os
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")


def load_data():
    data = pd.read_csv(DATA_PATH)

    data = data.drop(columns=['Unnamed: 32', 'id'], errors='ignore')
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    return data


def train_model(data):
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=10000, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model, scaler


def save_artifacts(model, scaler):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)


def main():
    data = load_data()
    model, scaler = train_model(data)
    save_artifacts(model, scaler)


if __name__ == "__main__":
    main()
