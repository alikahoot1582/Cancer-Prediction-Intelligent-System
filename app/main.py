import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from sklearn.metrics import accuracy_score

# -------------------- CONFIG --------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS = {
    "model": os.path.join(BASE_DIR, "model", "model.pkl"),
    "scaler": os.path.join(BASE_DIR, "model", "scaler.pkl"),
    "data": os.path.join(BASE_DIR, "data", "data.csv"),
    "css": os.path.join(BASE_DIR, "assets", "style.css"),
    "pdf": os.path.join(BASE_DIR, "app", "preventions.pdf")
}

# -------------------- LOADERS --------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['Unnamed: 32', 'id'], errors='ignore')
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df


@st.cache_resource
def load_model(model_path, scaler_path):
    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    return model, scaler


# -------------------- UI COMPONENTS --------------------
def sidebar_inputs(data):
    st.sidebar.header("Cell Nuclei Measurements")
    features = data.drop('diagnosis', axis=1)

    return {
        col: st.sidebar.slider(
            label=col,
            min_value=float(features[col].min()),
            max_value=float(features[col].max()),
            value=float(features[col].mean())
        ) for col in features.columns
    }


# -------------------- DATA PROCESSING --------------------
def prepare_input(input_dict, scaler, feature_order):
    arr = np.array([input_dict[col] for col in feature_order]).reshape(1, -1)
    return scaler.transform(arr)


# -------------------- VISUALIZATION --------------------
def radar_chart(input_dict):
    keys = [k for k in input_dict if "_mean" in k]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[input_dict[k] for k in keys],
        theta=[k.replace("_mean", "").title() for k in keys],
        fill='toself'
    ))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
    return fig


# -------------------- MODEL LOGIC --------------------
def make_prediction(input_dict, model, scaler, feature_order):
    scaled = prepare_input(input_dict, scaler, feature_order)
    pred = model.predict(scaled)[0]
    probs = model.predict_proba(scaled)[0]
    return pred, probs


def calculate_accuracy(model, scaler, data):
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']

    preds = model.predict(scaler.transform(X))
    return accuracy_score(y, preds) * 100


# -------------------- MAIN APP --------------------
# Logo path
LOGO_PATH = os.path.join(BASE_DIR, "app", "logo2.png")
def main():
    st.set_page_config(
        page_title="BioSync Lifeguard ❤️",
        page_icon="❤️",
        layout="wide"
    )

    # Load styles
    if os.path.exists(PATHS["css"]):
        with open(PATHS["css"]) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Load resources
    data = load_data(PATHS["data"])
    model, scaler = load_model(PATHS["model"], PATHS["scaler"])

    feature_order = data.drop('diagnosis', axis=1).columns
    inputs = sidebar_inputs(data)

    accuracy = calculate_accuracy(model, scaler, data)

    # Header
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=120)

    st.title("BioSync Lifeguard ❤️")
    st.write("Predicts tumor type: **Benign** or **Malignant** based on cell data.")

    col1, col2 = st.columns([3, 1])

    # Left panel
    with col1:
        st.plotly_chart(radar_chart(inputs), use_container_width=True)

    # Right panel
    with col2:
        st.subheader("Prediction")

        pred, probs = make_prediction(inputs, model, scaler, feature_order)

        if pred == 0:
            st.success("✅ Benign")
        else:
            st.error("⚠️ Malignant")

        st.write(f"Benign Probability: {probs[0]:.4f}")
        st.write(f"Malignant Probability: {probs[1]:.4f}")
        st.write(f"Model Accuracy: {accuracy:.2f}%")

        st.info("⚠️ Not a substitute for professional medical advice.")

        # PDF Download
        st.markdown("### 📘 Prevention Guide")

        if os.path.exists(PATHS["pdf"]):
            with open(PATHS["pdf"], "rb") as f:
                st.download_button(
                    label="📄 Download PDF",
                    data=f,
                    file_name="preventions.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("PDF not found.")

    # Footer
    st.markdown("---")
    st.markdown("### Made by Muhammad Ali Kahoot")


if __name__ == "__main__":
    main()
