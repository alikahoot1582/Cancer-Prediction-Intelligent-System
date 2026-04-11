import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from sklearn.metrics import accuracy_score

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")
CSS_PATH = os.path.join(BASE_DIR, "assets", "style.css")


# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data = data.drop(columns=['Unnamed: 32', 'id'], errors='ignore')
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


# Load model & scaler
@st.cache_resource
def load_model():
    model = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    return model, scaler


# Sidebar inputs
def add_sidebar(data):
    st.sidebar.header("Cell Nuclei Measurements")
    input_dict = {}

    for col in data.drop('diagnosis', axis=1).columns:
        input_dict[col] = st.sidebar.slider(
            col,
            float(data[col].min()),
            float(data[col].max()),
            float(data[col].mean())
        )

    return input_dict


# Prepare input
def prepare_input(input_dict, scaler, data):
    feature_order = data.drop('diagnosis', axis=1).columns
    input_array = np.array([input_dict[col] for col in feature_order]).reshape(1, -1)
    return scaler.transform(input_array)


# Radar chart
def plot_radar(input_dict):
    mean_keys = [k for k in input_dict if "_mean" in k]

    values = [input_dict[k] for k in mean_keys]
    categories = [k.replace("_mean", "").title() for k in mean_keys]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False
    )

    return fig


# Prediction
def predict(input_dict, model, scaler, data):
    input_scaled = prepare_input(input_dict, scaler, data)

    prediction = model.predict(input_scaled)[0]
    probs = model.predict_proba(input_scaled)[0]

    return prediction, probs


# Model accuracy
def get_model_accuracy(model, scaler, data):
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    acc = accuracy_score(y, y_pred)
    return acc * 100


# Main app
def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon="👩‍⚕️",
        layout="wide"
    )

    # Load CSS
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    data = load_data()
    model, scaler = load_model()

    input_dict = add_sidebar(data)
    accuracy = get_model_accuracy(model, scaler, data)

    st.title("Breast Cancer Predictor")
    st.write(
        "Predicts whether a tumor is **benign or malignant** "
        "based on cell measurements."
    )

    col1, col2 = st.columns([3, 1])

    # Left: Radar chart
    with col1:
        st.plotly_chart(plot_radar(input_dict), use_container_width=True)

    # Right: Prediction
    with col2:
        st.subheader("Prediction")

        prediction, probs = predict(input_dict, model, scaler, data)

        if prediction == 0:
            st.success("✅ Benign")
        else:
            st.error("⚠️ Malignant")

        st.write(f"Benign Probability: {probs[0]:.4f}")
        st.write(f"Malignant Probability: {probs[1]:.4f}")

        # Show accuracy
        st.write(f"Model Accuracy: {accuracy:.2f}%")

        st.info("⚠️ Not a substitute for professional medical diagnosis.")

    # Footer
    st.markdown("---")
    st.markdown("### Made by Muhammad Ali Kahoot")


if __name__ == "__main__":
    main()
