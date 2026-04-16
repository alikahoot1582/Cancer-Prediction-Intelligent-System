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
API_KEY = "your_groq_api_key_here"   # 👈 replace
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


# -------------------- UI INPUT --------------------
def sidebar_inputs(data):
    st.sidebar.header("Cell Measurements (Breast Tissue)")
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


# -------------------- PROCESS --------------------
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
def generate_ai_analysis(input_dict, prediction, probs, mode):
    try:
        if mode == "Basic":
            style = "Explain in very simple, short, patient-friendly language."
        else:
            style = "Give a detailed but still easy-to-understand explanation."

        prompt = f"""
        You are a compassionate breast cancer awareness assistant.

        Result: {"Malignant (higher risk)" if prediction == 1 else "Benign (lower risk)"}
        Probabilities: {probs}

        Patient data: {input_dict}

        {style}

        Include:
        - What this result means
        - Whether the risk seems low or higher
        - Practical next steps (doctor visit, screening, etc.)
        - General prevention tips for breast health
        - Reassurance without giving false certainty

        IMPORTANT:
        - Do NOT say this is a confirmed diagnosis
        - Encourage consulting a doctor
        - Keep tone calm and supportive
        """

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful and careful medical assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI Error: {e}"


# -------------------- MAIN --------------------
def main():
    st.set_page_config(
        page_title="Breast Cancer Risk Assistant",
        layout="wide"
    )

    # Load CSS
    if os.path.exists(PATHS["css"]):
        with open(PATHS["css"]) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Load
    data = load_data(PATHS["data"])
    model, scaler = load_model(PATHS["model"], PATHS["scaler"])

    feature_order = data.drop('diagnosis', axis=1).columns
    inputs = sidebar_inputs(data)
    accuracy = calculate_accuracy(model, scaler, data)

    # Header (NO LOGO)
    st.title("Breast Cancer Risk Assistant")
    st.write("This tool estimates whether a tumor is likely **benign or malignant** based on cell measurements.")

    # ⚠️ STRONG DISCLAIMER
    st.warning(
        "⚠️ This tool is for educational purposes only and is NOT a medical diagnosis. "
        "Always consult a qualified doctor for medical advice, diagnosis, or treatment."
    )

    col1, col2 = st.columns([3, 1])

    # LEFT
    with col1:
        st.plotly_chart(radar_chart(inputs), use_container_width=True)

    # RIGHT
    with col2:
        st.subheader("Prediction")

        pred, probs = make_prediction(inputs, model, scaler, feature_order)

        if pred == 0:
            st.success("Result: Likely Benign (Lower Risk)")
        else:
            st.error("Result: Possible Malignant (Higher Risk)")

        st.write(f"Benign Probability: {probs[0]:.3f}")
        st.write(f"Malignant Probability: {probs[1]:.3f}")
        st.write(f"Model Accuracy: {accuracy:.2f}%")

        # ---------------- AI MODE ----------------
        st.markdown("### 🤖 AI Analysis")

        mode = st.radio("Select Mode:", ["Basic", "Detailed"])

        if st.button("Generate AI Analysis"):
            with st.spinner("Analyzing your results..."):
                analysis = generate_ai_analysis(inputs, pred, probs, mode)
                st.write(analysis)

        # ---------------- PDF ----------------
        st.markdown("### 📘 Prevention Guide")

        if os.path.exists(PATHS["pdf"]):
            with open(PATHS["pdf"], "rb") as f:
                st.download_button(
                    "Download Prevention Guide",
                    f,
                    file_name="breast_cancer_prevention.pdf"
                )
        else:
            st.info("Prevention guide not available.")

    st.markdown("---")
    st.caption("Made for awareness and educational support only.")


if __name__ == "__main__":
    main()
