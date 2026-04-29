import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from sklearn.metrics import accuracy_score
from groq import Groq

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATHS = {
    "model": os.path.join(BASE_DIR, "model", "model.pkl"),
    "scaler": os.path.join(BASE_DIR, "model", "scaler.pkl"),
    "data": os.path.join(BASE_DIR, "data", "data.csv"),
    "pdf": os.path.join(BASE_DIR, "app", "preventions.pdf")
}

API_KEY = "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" 
client = Groq(api_key=API_KEY)

# ---------------- LOADERS ----------------
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


# ---------------- INPUT ----------------
def sidebar_inputs(data):
    st.sidebar.header("Cell Measurements")
    features = data.drop('diagnosis', axis=1)

    return {
        col: st.sidebar.slider(
            col,
            float(features[col].min()),
            float(features[col].max()),
            float(features[col].mean())
        ) for col in features.columns
    }


# ---------------- MODEL ----------------
def prepare_input(input_dict, scaler, feature_order):
    arr = np.array([input_dict[col] for col in feature_order]).reshape(1, -1)
    return scaler.transform(arr)


def make_prediction(input_dict, model, scaler, feature_order):
    scaled = prepare_input(input_dict, scaler, feature_order)
    return model.predict(scaled)[0], model.predict_proba(scaled)[0]


def calculate_accuracy(model, scaler, data):
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis']
    return accuracy_score(y, model.predict(scaler.transform(X))) * 100


# ---------------- CHART ----------------
def radar_chart(input_dict):
    keys = [k for k in input_dict if "_mean" in k]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[input_dict[k] for k in keys],
        theta=[k.replace("_mean", "").title() for k in keys],
        fill='toself'
    ))

    return fig


# ---------------- AI ----------------
def generate_initial_analysis(inputs, pred, probs, mode):
    style = "simple" if mode == "Basic" else "detailed"

    prompt = f"""
    Prediction: {'Malignant' if pred else 'Benign'}
    Probabilities: {probs}

    Explain in a {style} way.
    Include meaning, risk, next steps, and prevention.
    Say clearly this is NOT a diagnosis.
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    return response.choices[0].message.content


def chat_reply(messages):
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            *messages
        ],
        temperature=0.5
    )

    return response.choices[0].message.content


# ---------------- MAIN ----------------
def main():
    st.set_page_config(page_title="Breast Cancer Assistant", page_icon="🩺", layout="wide")

    data = load_data(PATHS["data"])
    model, scaler = load_model(PATHS["model"], PATHS["scaler"])

    feature_order = data.drop('diagnosis', axis=1).columns
    inputs = sidebar_inputs(data)
    accuracy = calculate_accuracy(model, scaler, data)

    st.title("🩺 Breast Cancer Risk Assistant")
    st.caption("AI-powered model that estimates whether a tumor is benign or malignant based on cell measurements.")
    st.warning("Not a medical diagnosis. Consult a doctor.")

    # Chart
    st.subheader("Cell Overview")
    st.plotly_chart(radar_chart(inputs), use_container_width=True)

    # Prediction
    pred, probs = make_prediction(inputs, model, scaler, feature_order)

    col1, col2, col3 = st.columns(3)
    col1.metric("Result", "Benign" if pred == 0 else "Malignant")
    col2.metric("Malignant %", f"{probs[1]:.2f}")
    col3.metric("Accuracy", f"{accuracy:.2f}%")

    # ---------------- AI ANALYSIS ----------------
    st.markdown("---")
    st.subheader("🤖 AI Analysis")

    mode = st.radio("Mode", ["Basic", "Detailed"], horizontal=True)

    if "chat" not in st.session_state:
        st.session_state.chat = []

    # Initial analysis (ONLY ONCE)
    if st.button("Generate Analysis"):
        analysis = generate_initial_analysis(inputs, pred, probs, mode)
        st.session_state.chat = [{"role": "assistant", "content": analysis}]

    # Show chat
    for msg in st.session_state.chat:
        if msg["role"] == "assistant":
            st.info(msg["content"])
        else:
            st.write(f"You: {msg['content']}")

    # User input
    user_input = st.text_input("Ask a follow-up question")

    if user_input:
        st.session_state.chat.append({"role": "user", "content": user_input})

        reply = chat_reply(st.session_state.chat)

        st.session_state.chat.append({"role": "assistant", "content": reply})
        st.rerun()

    # FAQ
    st.markdown("---")
    st.subheader("FAQ")

    with st.expander("Is this a diagnosis?"):
        st.write("No, it's only a prediction.")

    with st.expander("How accurate is this?"):
        st.write("Model accuracy is high but not perfect.")

    with st.expander("How accurate is this model?"):
        st.write(
            "The model provides an estimate based on historical data and patterns. "
            "While it may show high accuracy, it is not perfect and should not be relied "
            "on for medical decisions. Always consult a healthcare professional."
        )

    with st.expander("What should I do next?"):
        st.write("Consult a doctor for proper testing.")

    # PDF
    if os.path.exists(PATHS["pdf"]):
        with open(PATHS["pdf"], "rb") as f:
            st.download_button("Download Guide", f, "guide.pdf")


if __name__ == "__main__":
    main()
