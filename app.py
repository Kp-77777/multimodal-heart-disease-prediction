import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import tempfile
from video_model.inference import load_echo_model, predict_video

from multimodal.tabular_ensemble import ensemble_tabular
from multimodal.late_fusion import late_fusion
from video_model.inference import predict_video
from explainability.shap_tabular import explain_tabular, explain_tabular_table
from explainability.gradcam_video import gradcam_for_video


@st.cache_resource
def load_echo_cached():
    return load_echo_model()

echo_model = load_echo_cached()

if hasattr(st, "fragment"):
    fragment = st.fragment
else:
    def fragment(func):
        return func

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Heart Disease Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# Professional Styling
# ===============================
st.markdown("""
<style>
    /* Main background and font */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
        font-weight: 600;
    }
    
    /* Custom titles */
    .main-title {
        font-size: 38px;
        font-weight: 700;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 8px;
    }
    .subtitle {
        font-size: 18px;
        color: #A0AEC0;
        text-align: center;
        margin-bottom: 40px;
        font-weight: 400;
    }
    
    /* Section headers */
    .section-header {
        font-size: 26px;
        font-weight: 600;
        color: #FFFFFF;
        border-bottom: 2px solid #3182CE;
        padding-bottom: 8px;
        margin-top: 40px;
        margin-bottom: 20px;
    }
    
    /* Cards for predictions */
    .pred-card {
        background-color: #1A202C;
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
        border-left: 4px solid #3182CE;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .model-name {
        font-size: 18px;
        font-weight: 600;
        color: #CBD5E0;
        margin-bottom: 8px;
    }
    .prob-text {
        font-size: 16px;
        color: #E2E8F0;
        margin: 6px 0;
    }
    .risk-high {
        color: #FF5252;
        font-weight: 700;
        font-size: 18px;
    }
    .risk-low {
        color: #48BB78;
        font-weight: 700;
        font-size: 18px;
    }
    .risk-medium {
        color: #F6C244;
        font-weight: 700;
        font-size: 18px;
    }
    
    /* Highest risk highlight */
    .final-risk {
        background-color: #2D3748;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        margin-top: 20px;
        border: 1px solid #4A5568;
    }
    
    /* Sidebar styling */
    .css-1d391kg {  /* Sidebar */
        background-color: #161B22;
    }
    .css-1v3fvz1 {  /* Radio buttons in sidebar */
        color: #E2E8F0;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #3182CE;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #2B6CB0;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# Session State
# ===============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ===============================
# Header
# ===============================
st.markdown("<div class='main-title'>Heart Disease Prediction System</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Clinical decision support using patient risk factors and echocardiography analysis</div>",
    unsafe_allow_html=True
)

# ===============================
# Sidebar
# ===============================
st.sidebar.header("📊 Prediction Mode")
mode = st.sidebar.radio(
    "Select prediction method",
    ["Tabular Risk Factors", "ECHO Video Analysis", "Multimodal Prediction", "Chat Assistant"]
)

# ===============================
# Load Tabular Models
# ===============================
@st.cache_resource
def load_tabular_models():
    model_paths = {
        "Logistic Regression": "utils/Logistic Regression_model.pkl",
        "XGBoost": "utils/XGBoost_model.pkl",
        "Random Forest": "utils/Random Forest_model.pkl",
        "SVM": "utils/SVM_model.pkl"
    }
    models = {}
    for name, path in model_paths.items():
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
    return models

models = load_tabular_models()
training_columns = joblib.load("utils/training_columns.pkl")

# ===============================
# TABULAR MODE
# ===============================
if mode == "Tabular Risk Factors":
    st.markdown("<div class='section-header'>📋 Patient Clinical Data</div>", unsafe_allow_html=True)
    st.markdown("Enter patient information to assess cardiovascular risk.")

    with st.form("tabular_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=50)
            sex = st.selectbox("Sex", options=["M", "F"], index=0)
            chest_pain = st.selectbox("Chest Pain Type", options=["ASY", "ATA", "NAP", "TA"])
            resting_bp = st.number_input("Resting Blood Pressure (mmHg)", min_value=0, max_value=300, value=140)
            cholesterol = st.number_input("Serum Cholesterol (mg/dL)", min_value=0, max_value=600, value=200)
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[0, 1], format_func=lambda x: "Yes" if x else "No")

        with col2:
            resting_ecg = st.selectbox("Resting ECG Result", options=["Normal", "ST", "LVH"])
            max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=220, value=150)
            exercise_angina = st.selectbox("Exercise-Induced Angina", options=["N", "Y"], format_func=lambda x: "Yes" if x == "Y" else "No")
            oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=-3.0, max_value=7.0, value=1.0, step=0.1)
            st_slope = st.selectbox("ST Slope", options=["Up", "Flat", "Down"])

        submitted = st.form_submit_button("🔬 Run Risk Assessment", use_container_width=True)

    if submitted:
        # Prepare input
        input_df = pd.DataFrame([{
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest_pain,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "RestingECG": resting_ecg,
            "MaxHR": max_hr,
            "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak,
            "ST_Slope": st_slope
        }])

        encoded = pd.get_dummies(input_df)
        encoded = encoded.reindex(columns=training_columns, fill_value=0)

        st.markdown("<div class='section-header'>📈 Model Predictions</div>", unsafe_allow_html=True)
        st.info("*Most reliable model- XGBOOST (Acc- 0.891304       F1- 0.902913*")

        best_prob = -1
        best_model_name = None

        for name, model in models.items():
            prob = model.predict_proba(encoded)[0][1]
            healthy_prob = 1 - prob

            risk_class = "risk-high" if prob > 0.45 else "risk-low"
            risk_text = "High Risk of Heart Disease" if prob > 0.45 else "Low Risk"

            with st.container():
                st.markdown(f"""
                <div class='pred-card'>
                    <div class='model-name'>{name}</div>
                    <div class='prob-text'>Disease Probability: <strong>{prob*100:.2f}%</strong></div>
                    <div class='prob-text'>Healthy Probability: <strong>{healthy_prob*100:.2f}%</strong></div>
                    <div style='margin-top:12px; font-size:18px;'>
                        Assessment: <span class='{risk_class}'>{risk_text}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if prob > best_prob:
                best_prob = prob
                best_model_name = name

        st.markdown("---")
        st.markdown(f"""
        <div class='final-risk'>
            <strong>Highest Estimated Risk: {best_prob*100:.2f}%</strong><br>
            <small>From model: <strong>{best_model_name}</strong></small>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>🧩 Input Feature Summary</div>", unsafe_allow_html=True)
        st.dataframe(input_df)

        # Tabular explainability
        with st.expander("🧭 Explainability: Tabular SHAP"):
            st.markdown("<div class='section-header'>SHAP Feature Importance</div>", unsafe_allow_html=True)
            try:
                with st.spinner("Generating SHAP explanations..."):
                    shap_fig = explain_tabular(models, encoded, nsamples=50)
                    st.pyplot(shap_fig)
                    shap_table = explain_tabular_table(models, encoded, nsamples=50)
                    st.dataframe(shap_table)
            except Exception as e:
                st.error(f"SHAP explanation failed: {e}")

# ===============================
# ECHO VIDEO MODE
# ===============================
elif mode == "ECHO Video Analysis":
    st.markdown("<div class='section-header'>🫀 Echocardiography Video Analysis</div>", unsafe_allow_html=True)
    st.markdown("Upload an apical 4-chamber echocardiogram video for automated cardiac function assessment.")

    uploaded_video = st.file_uploader("Choose an ECHO video file (.mp4 recommended)", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        st.video(uploaded_video)
        
        if st.button("🔬 Analyze Video", use_container_width=True):
            with st.spinner("Processing video and running EchoNet-Dynamic inference..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(uploaded_video.read())
                    video_path = tmp.name

                prob = predict_video(video_path, echo_model)

            st.markdown("<div class='section-header'>📊 Echo Analysis Result</div>", unsafe_allow_html=True)
            
            risk_class = "risk-high" if prob > 0.45 else "risk-low"
            risk_text = "High Risk based on echocardiography" if prob > 0.45 else "Low Risk based on echocardiography"

            st.markdown(f"""
            <div class='pred-card'>
                <div class='model-name'>Model Prediction</div>
                <div class='prob-text'>Probability of abnormal cardiac function: <strong>{prob*100:.2f}%</strong></div>
                <div style='margin-top:12px; font-size:18px;'>
                    Assessment: <span class='{risk_class}'>{risk_text}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🧭 Explainability: Echo Grad-CAM"):
                st.subheader("Best Frame with Convergent Heatmap")
                try:
                    cam_img = gradcam_for_video(video_path, echo_model)
                    st.image(cam_img, use_container_width=True)
                except Exception as e:
                    st.error(f"Grad-CAM image failed: {e}")

# ===============================
# MULTIMODAL PLACEHOLDER
# ===============================
elif mode == "Multimodal Prediction":
    st.markdown("<div class='section-header'>🔗 Multimodal Heart Disease Assessment</div>", unsafe_allow_html=True)
    st.markdown(
        "This module combines **clinical risk factors** and **echocardiography video analysis** "
        "using a **late-fusion, probability-level ensemble**."
    )

    with st.form("multimodal_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=50)
            sex = st.selectbox("Sex", ["M", "F"])
            chest_pain = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
            resting_bp = st.number_input("Resting Blood Pressure (mmHg)", 0, 300, 140)
            cholesterol = st.number_input("Serum Cholesterol (mg/dL)", 0, 600, 200)
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

        with col2:
            resting_ecg = st.selectbox("Resting ECG Result", ["Normal", "ST", "LVH"])
            max_hr = st.number_input("Maximum Heart Rate Achieved", 0, 220, 150)
            exercise_angina = st.selectbox("Exercise-Induced Angina", ["N", "Y"])
            oldpeak = st.number_input("Oldpeak (ST Depression)", -3.0, 7.0, 1.0)
            st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

        uploaded_video = st.file_uploader(
            "Upload Echocardiography Video (.mp4)", type=["mp4", "avi", "mov"]
        )
        if uploaded_video is not None:
            st.video(uploaded_video)
        submitted = st.form_submit_button("🔬 Run Multimodal Assessment", use_container_width=True)

    if submitted:
        if uploaded_video is None:
            st.error("Please upload an echocardiography video for multimodal assessment.")
        else:
            # ------------------------------
            # TABULAR INPUT PREPARATION
            # ------------------------------
            input_df = pd.DataFrame([{
                "Age": age,
                "Sex": sex,
                "ChestPainType": chest_pain,
                "RestingBP": resting_bp,
                "Cholesterol": cholesterol,
                "FastingBS": fasting_bs,
                "RestingECG": resting_ecg,
                "MaxHR": max_hr,
                "ExerciseAngina": exercise_angina,
                "Oldpeak": oldpeak,
                "ST_Slope": st_slope
            }])

            encoded = pd.get_dummies(input_df)
            encoded = encoded.reindex(columns=training_columns, fill_value=0)

            # ------------------------------
            # TABULAR ENSEMBLE (PERFORMANCE-WEIGHTED)
            # ------------------------------
            P_tab, tab_details = ensemble_tabular(models, encoded)

            # ------------------------------
            # ECHO MODEL INFERENCE
            # ------------------------------
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_video.read())
                video_path = tmp.name

            P_echo = predict_video(video_path, echo_model)

            # ------------------------------
            # LATE FUSION (CLINICALLY WEIGHTED)
            # ------------------------------
            P_multi = late_fusion(P_tab, P_echo, alpha=0.6)
            P_multi_cat = late_fusion(P_tab, P_echo, alpha=0.6, as_category=True)

            # ------------------------------
            # RESULTS
            # ------------------------------
            st.markdown("<div class='section-header'>📊 Multimodal Result</div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class='pred-card'>
                <div class='model-name'>Tabular Risk Ensemble</div>
                <div class='prob-text'>Probability of Heart Failure: <strong>{P_tab*100:.2f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class='pred-card'>
                <div class='model-name'>Echocardiography Model</div>
                <div class='prob-text'>Probability of disease due to Reduced EF: <strong>{P_echo*100:.2f}%</strong></div>
            </div>
            """, unsafe_allow_html=True)

            # Map category to styling and human-readable text
            if P_multi_cat == "low":
                risk_class = "risk-low"
                risk_text = "Low Risk of Heart Failure"
            elif P_multi_cat == "medium":
                risk_class = "risk-medium"
                risk_text = "Moderate Risk of Heart Failure"
            else:
                risk_class = "risk-high"
                risk_text = "High Risk of Heart Failure"

            st.markdown(f"""
            <div class='final-risk'>
                <strong>Final Multimodal Probability: {P_multi*100:.2f}%</strong><br>
                <span class='{risk_class}'>{risk_text} ({P_multi_cat.title()})</span><br>
                <small>Fusion Strategy: Late fusion (Echo 60% • Clinical 40%)</small>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🧭 Explainability: SHAP + Grad-CAM"):
                st.subheader("Tabular SHAP Waterfall Plot")
                try:
                    with st.spinner("Generating SHAP waterfall plot..."):
                        shap_fig = explain_tabular(models, encoded, nsamples=50)
                        st.pyplot(shap_fig)
                except Exception as e:
                    st.error(f"SHAP waterfall plot failed: {e}")

                st.subheader("Tabular SHAP Table")
                try:
                    with st.spinner("Generating SHAP explanations..."):
                        shap_table = explain_tabular_table(models, encoded, nsamples=50)
                        st.dataframe(shap_table)
                except Exception as e:
                    st.error(f"SHAP explanation failed: {e}")

                st.subheader("Best Frame with Convergent Heatmap")
                try:
                    cam_img = gradcam_for_video(video_path, echo_model)
                    st.image(cam_img, use_container_width=True)
                except Exception as e:
                    st.error(f"Grad-CAM failed: {e}")


# ===============================
# CHAT ASSISTANT
# ===============================
else:  # Chat Assistant
    st.markdown("<div class='section-header'>💬 Clinical Chat Assistant</div>", unsafe_allow_html=True)
    st.markdown("Ask questions about cardiovascular risk, interpretation of results, or guideline recommendations.")

    user_input = st.chat_input("Type your question here...")

    if user_input:
        st.session_state.chat_history.append({"user": user_input, "bot": "This assistant will be connected to a specialized medical LLM in the next phase."})

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("assistant"):
            st.write(chat["bot"])