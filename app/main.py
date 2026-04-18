import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from sklearn.metrics import accuracy_score

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Breast Cancer Assistant",
    page_icon="🩺",
    layout="wide"
)

# ---------------- STYLING ----------------
st.markdown("""
<style>
body {
    background-color: #f4f7fb;
}
.block-container {
    padding-top: 2rem;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}
.title {
    font-size: 30px;
    font-weight: 700;
    color: #1f3b4d;
}
.subtitle {
    color: #6c757d;
    margin-bottom: 10px;
}
.result-good {
    color: #28a745;
    font-size: 26px;
    font-weight: bold;
}
.result-bad {
    color: #dc3545;
    font-size: 26px;
    font-weight: bold;
}
.section-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "model": os.path.join(BASE_DIR, "model.pkl"),
    "scaler": os.path.join(BASE_DIR, "scaler.pkl"),
    "data": os.path.join(BASE_DIR, "data.csv")
}

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

# ---------------- INPUT UI ----------------
def sidebar_inputs(data):
    st.sidebar.markdown("## 🧾 Patient Measurements")

    features = data.drop('diagnosis', axis=1)

    input_dict = {}

    # Grouping important features only (clean UI)
    with st.sidebar.expander("🔬 Cell Size"):
        for col in features.columns[:5]:
            input_dict[col] = st.slider(col, float(features[col].min()), float(features[col].max()), float(features[col].mean()))

    with st.sidebar.expander("📏 Texture & Shape"):
        for col in features.columns[5:10]:
            input_dict[col] = st.slider(col, float(features[col].min()), float(features[col].max()), float(features[col].mean()))

    with st.sidebar.expander("🧪 Smoothness & Compactness"):
        for col in features.columns[10:15]:
            input_dict[col] = st.slider(col, float(features[col].min()), float(features[col].max()), float(features[col].mean()))

    # Remaining features hidden but still used
    for col in features.columns[15:]:
        input_dict[col] = features[col].mean()

    return input_dict

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
    keys = list(input_dict.keys())[:10]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[input_dict[k] for k in keys],
        theta=[k.replace("_mean", "").title() for k in keys],
        fill='toself'
    ))

    fig.update_layout(
        polar=dict(bgcolor="#f4f7fb"),
        showlegend=False
    )

    return fig

# ---------------- MAIN ----------------
def main():
    # Load
    data = load_data(PATHS["data"])
    model, scaler = load_model(PATHS["model"], PATHS["scaler"])

    feature_order = data.drop('diagnosis', axis=1).columns
    inputs = sidebar_inputs(data)
    accuracy = calculate_accuracy(model, scaler, data)

    # Header
    st.markdown('<div class="title">🩺 Breast Cancer Risk Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered screening support tool</div>', unsafe_allow_html=True)
    st.warning("This tool is for educational purposes only. Always consult a doctor.")

    # Prediction
    pred, probs = make_prediction(inputs, model, scaler, feature_order)

    col1, col2 = st.columns([1, 2])

    # ---------------- RESULT CARD ----------------
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Result Summary</div>', unsafe_allow_html=True)

        if pred == 0:
            st.markdown('<div class="result-good">✅ Benign</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-bad">⚠️ Malignant</div>', unsafe_allow_html=True)

        st.progress(float(probs[1]))
        st.write(f"Risk Level: {probs[1]*100:.2f}%")

        st.write(f"Model Accuracy: {accuracy:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- CHART ----------------
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 Cell Analysis</div>', unsafe_allow_html=True)
        st.plotly_chart(radar_chart(inputs), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- NEXT STEPS ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧾 Recommended Next Steps</div>', unsafe_allow_html=True)

    if pred == 1:
        st.error("Consult a certified oncologist immediately for further diagnostic tests.")
    else:
        st.success("Maintain regular screenings and a healthy lifestyle.")

    st.write("""
    • Schedule a medical check-up  
    • Do not rely only on this tool  
    • Follow professional medical advice  
    """)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- FAQ ----------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">❓ FAQ</div>', unsafe_allow_html=True)

    with st.expander("Is this a diagnosis?"):
        st.write("No. This is only an AI-based prediction.")

    with st.expander("How accurate is this?"):
        st.write("The model is trained on historical data but is not 100% reliable.")

    with st.expander("What should I do next?"):
        st.write("Always consult a qualified doctor for medical decisions.")

    st.markdown('</div>', unsafe_allow_html=True)


# ---------------- RUN ----------------
if __name__ == "__main__":
    main()
