import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "features.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")
CSS_PATH = os.path.join(BASE_DIR, "assets", "style.css")


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data = data.drop(columns=['Unnamed: 32', 'id'], errors='ignore')
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


@st.cache_resource
def load_model():
    model = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    features = pickle.load(open(FEATURES_PATH, "rb"))
    return model, scaler, features


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


def prepare_input(input_dict, scaler, features):
    input_array = np.array([input_dict[f] for f in features]).reshape(1, -1)
    return scaler.transform(input_array)


def plot_radar(input_dict):
    mean_keys = [k for k in input_dict if "_mean" in k]

    values = [input_dict[k] for k in mean_keys]
    categories = [k.replace("_mean", "").title() for k in mean_keys]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Mean Features'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True
    )

    return fig


def predict(input_dict, model, scaler, features):
    input_scaled = prepare_input(input_dict, scaler, features)

    prediction = model.predict(input_scaled)[0]
    probs = model.predict_proba(input_scaled)[0]

    return prediction, probs


def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon="👩‍⚕️",
        layout="wide"
    )

    # Load CSS safely
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    data = load_data()
    model, scaler, features = load_model()

    input_dict = add_sidebar(data)

    st.title("Breast Cancer Predictor")
    st.write(
        "This app predicts whether a tumor is **benign or malignant** "
        "based on cell nuclei measurements."
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        st.plotly_chart(plot_radar(input_dict), use_container_width=True)

    with col2:
        st.subheader("Prediction")

        prediction, probs = predict(input_dict, model, scaler, features)

        if prediction == 0:
            st.success("✅ Benign")
        else:
            st.error("⚠️ Malignant")

        st.write(f"Benign Probability: {probs[0]:.4f}")
        st.write(f"Malignant Probability: {probs[1]:.4f}")

        st.info(
            "⚠️ This tool assists medical analysis but is NOT a replacement "
            "for professional diagnosis."
        )


if __name__ == "__main__":
    main()
