import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from sklearn.metrics import accuracy_score
from groq import Groq

# -------------------- CONFIG --------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS = {
    "model": os.path.join(BASE_DIR, "model", "model.pkl"),
    "scaler": os.path.join(BASE_DIR, "model", "scaler.pkl"),
    "data": os.path.join(BASE_DIR, "data", "data.csv"),
    "css": os.path.join(BASE_DIR, "assets", "style.css"),
    "pdf": os.path.join(BASE_DIR, "app", "preventions.pdf")
}

# -------------------- GROQ --------------------
API_KEY = "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"
client = Groq(api_key=API_KEY)

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


# -------------------- SIDEBAR INPUT --------------------
def sidebar_inputs(data):
    st.sidebar.header("Cell Measurements")
    features = data.drop('diagnosis', axis=1)

    inputs = {}
    for col in features.columns:
        inputs[col] = st.sidebar.slider(
            col,
            float(features[col].min()),
            float(features[col].max()),
            float(features[col].mean())
        )
    return inputs


# -------------------- PREP --------------------
def prepare_input(input_dict, scaler, feature_order):
    arr = np.array([input_dict[col] for col in feature_order]).reshape(1, -1)
    return scaler.transform(arr)


# -------------------- CHART --------------------
def radar_chart(input_dict):
    keys = [k for k in input_dict if "_mean" in k]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[input_dict[k] for k in keys],
        theta=[k.replace("_mean", "").title() for k in keys],
        fill='toself'
    ))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True)))
    return fig


# -------------------- MODEL --------------------
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


# -------------------- AI ANALYSIS --------------------
def generate_ai_analysis(input_dict, prediction, probs, mode, chat_history=None):
    style = "simple and short" if mode == "Basic" else "detailed but clear"

    base_prompt = f"""
    You are a careful medical assistant.

    Prediction: {"Malignant" if prediction else "Benign"}
    Probabilities: {probs}

    Explain in a {style} way.

    Include meaning, risk level, next steps, and prevention.
    Always remind this is NOT a diagnosis.
    """

    messages = [{"role": "system", "content": base_prompt}]

    if chat_history:
        messages.extend(chat_history)

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.5
    )

    return response.choices[0].message.content


# -------------------- MAIN --------------------
def main():
    st.set_page_config(page_title="Breast Cancer Assistant", page_icon="🩺", layout="wide")

    # Load CSS
    if os.path.exists(PATHS["css"]):
        with open(PATHS["css"]) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    data = load_data(PATHS["data"])
    model, scaler = load_model(PATHS["model"], PATHS["scaler"])

    feature_order = data.drop('diagnosis', axis=1).columns
    inputs = sidebar_inputs(data)
    accuracy = calculate_accuracy(model, scaler, data)

    # Header
    st.title("🩺 Breast Cancer Risk Assistant")
    st.caption("AI-powered educational tool for breast health awareness")

    st.warning("This is NOT a medical diagnosis. Always consult a doctor.")

    # Chart
    st.subheader("Cell Measurement Overview")
    st.plotly_chart(radar_chart(inputs), use_container_width=True)

    # Prediction
    pred, probs = make_prediction(inputs, model, scaler, feature_order)

    st.subheader("Prediction")
    col1, col2, col3 = st.columns(3)

    col1.metric("Result", "Benign" if pred == 0 else "Malignant")
    col2.metric("Benign %", f"{probs[0]:.2f}")
    col3.metric("Accuracy", f"{accuracy:.2f}%")

    # Chat system
    st.markdown("---")
    st.subheader("🤖 AI Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    mode = st.radio("Mode", ["Basic", "Detailed"], horizontal=True)

    if st.button("Generate Analysis"):
        reply = generate_ai_analysis(inputs, pred, probs, mode)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            st.info(msg["content"])
        else:
            st.write(f"You: {msg['content']}")

    user_input = st.text_input("Ask something...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        reply = generate_ai_analysis(inputs, pred, probs, mode, st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

    # FAQ
    st.markdown("---")
    st.subheader("FAQ")

    with st.expander("Is this a diagnosis?"):
        st.write("No, only a prediction model.")

    with st.expander("Should I worry?"):
        st.write("Consult a doctor for proper evaluation.")

    with st.expander("What next?"):
        st.write("Medical screening like mammogram or biopsy.")

    # PDF
    st.markdown("---")
    if os.path.exists(PATHS["pdf"]):
        with open(PATHS["pdf"], "rb") as f:
            st.download_button("Download Prevention Guide", f, "guide.pdf")

    st.caption("Educational use only")


if __name__ == "__main__":
    main()
