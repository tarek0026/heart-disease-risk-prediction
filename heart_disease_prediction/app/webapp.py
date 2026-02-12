import streamlit as st
import joblib
import pandas as pd

# --------------------------
# Load model
# --------------------------
model = joblib.load("heart_disease_prediction/Model/heart_disease_model.pkl")

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# --------------------------
# Title
# --------------------------
st.markdown(
    "<h1 style='text-align:center;'>❤️ Heart Health Survey ❤️</h1>",
    unsafe_allow_html=True
)

st.write("### Please answer all questions before prediction.")
st.divider()

# --------------------------
# Inputs
# --------------------------

# Age (Keyboard input)
age = st.number_input(
    "🧍 Age (years)",
    min_value=1,
    max_value=120,
    value=None,
    placeholder="Type your age..."
)

# Gender
st.markdown("👤 Gender (click one):")
sex_text = st.radio(
    "Gender",
    ["Male", "Female"],
    index=None,
    key="sex",
    label_visibility="collapsed"
)

# Chest Pain Type
st.markdown("💢 Chest Pain Type (choose one):")
cp_text = st.radio(
    "Chest Pain Type",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-Anginal Pain",
        "Asymptomatic"
    ],
    index=None,
    key="cp",
    label_visibility="collapsed"
)

# Sliders for numeric values
trestbps = st.slider("🩺 Resting Blood Pressure (mm Hg)", 80, 250, value=120)
chol = st.slider("🧪 Cholesterol Level (mg/dL)", 50, 600, value=200)
thalch = st.slider("❤️ Maximum Heart Rate Achieved", 50, 250, value=150)
oldpeak = st.slider("📉 Oldpeak (ST Depression)", 0.0, 10.0, value=1.0)

# Fasting Blood Sugar
st.markdown("🍬 Fasting Blood Sugar > 120? (click one):")
fbs_text = st.radio(
    "Fasting Blood Sugar",
    ["Yes", "No"],
    index=None,
    key="fbs",
    label_visibility="collapsed"
)

# Exercise Angina
st.markdown("🏃 Exercise Induced Angina? (click one):")
exang_text = st.radio(
    "Exercise Angina",
    ["Yes", "No"],
    index=None,
    key="exang",
    label_visibility="collapsed"
)

# Rest ECG
st.markdown("📈 Resting ECG Result (choose one):")
restecg_text = st.radio(
    "Rest ECG",
    [
        "Normal",
        "ST-T Wave Abnormality",
        "Left Ventricular Hypertrophy"
    ],
    index=None,
    key="restecg",
    label_visibility="collapsed"
)

# Slope
st.markdown("📊 Slope of Peak Exercise ST Segment (choose one):")
slope_text = st.radio(
    "Slope",
    ["Upsloping", "Flat", "Downsloping"],
    index=None,
    key="slope",
    label_visibility="collapsed"
)

# ca
ca = st.selectbox(
    "🫀 Number of Major Vessels Colored (0–3):",
    [0, 1, 2, 3],
    key="ca"
)

# Thalassemia
st.markdown("🧬 Thalassemia Type (choose one):")
thal_text = st.radio(
    "Thalassemia",
    ["Normal", "Fixed Defect", "Reversible Defect"],
    index=None,
    key="thal",
    label_visibility="collapsed"
)

st.divider()

# --------------------------
# Prediction Button
# --------------------------
if st.button("🔍 Predict Heart Disease Risk", use_container_width=True):

    missing = []

    if age is None:
        missing.append("Age")

    if sex_text is None:
        missing.append("Gender")

    if cp_text is None:
        missing.append("Chest Pain Type")

    if fbs_text is None:
        missing.append("Fasting Blood Sugar")

    if exang_text is None:
        missing.append("Exercise Angina")

    if restecg_text is None:
        missing.append("Rest ECG Result")

    if slope_text is None:
        missing.append("Slope")

    if thal_text is None:
        missing.append("Thalassemia Type")

    # Missing fields warning
    if missing:
        st.error("⚠️ Please complete the following fields first:")
        for item in missing:
            st.write(f"❌ {item}")
        st.stop()

    # --------------------------
    # Encoding
    # --------------------------
    sex = 1 if sex_text == "Male" else 0

    cp_map = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-Anginal Pain": 2,
        "Asymptomatic": 3
    }
    cp = cp_map[cp_text]

    fbs = 1 if fbs_text == "Yes" else 0
    exang = 1 if exang_text == "Yes" else 0

    restecg_map = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }
    restecg = restecg_map[restecg_text]

    slope_map = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }
    slope = slope_map[slope_text]

    thal_map = {
        "Normal": 0,
        "Fixed Defect": 1,
        "Reversible Defect": 2
    }
    thal = thal_map[thal_text]

    # --------------------------
    # Prepare Data
    # --------------------------
    sample = {
        "age": age,
        "trestbps": trestbps,
        "chol": chol,
        "thalch": thalch,
        "oldpeak": oldpeak,
        "sex": sex,
        "cp": cp,
        "fbs": fbs,
        "restecg": restecg,
        "exang": exang,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    df = pd.DataFrame([sample])

    # --------------------------
    # Prediction
    # --------------------------
    prob = model.predict_proba(df)[0][1]
    pred = model.predict(df)[0]

    st.divider()

    if pred == 1:
        st.error(f"⚠️ High Risk of Heart Disease\n\nProbability = {prob:.2f}")
        st.write("👉 Please consult a doctor for medical advice.")
    else:
        st.success(f"✅ Low Risk Detected\n\nProbability = {prob:.2f}")
        st.write("💚 Keep maintaining a healthy lifestyle!")
