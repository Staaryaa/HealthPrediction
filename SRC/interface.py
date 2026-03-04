import streamlit as st
import numpy as np
import pandas as pd
import keras
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
import predict

SRC_PATH = Path(__file__).resolve().parent 
ROOT_PATH = SRC_PATH.parent
PRO_DATA_PATH = ROOT_PATH / "DATA" / "processed"
MODEL_PATH = ROOT_PATH/ "MODEL"
FINAL_MODEL_PATH = MODEL_PATH/"health_model.keras"

@st.cache_resource
def load_model_once():
    return keras.models.load_model(FINAL_MODEL_PATH, compile=False)

my_model = load_model_once()

st.title("HEALTH PREDICTOR SYSTEM")

st.sidebar.header("Patient Input")

age = int(st.sidebar.number_input("Age", 18, 100, 25))
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 22.5)
s_bp = int(st.sidebar.number_input("Systolic Blood Pressure", 80 , 200, 120))
d_bp = int(st.sidebar.number_input("Diastolic Blood Pressure", 40, 130, 80))
stress = st.sidebar.slider("Stress Levels", 0,10,5)

gender = st.sidebar.radio("Gender", ["Male", "Female", "Other"])
smoke = st.sidebar.radio("Smoking Habits", ["Never", "Former", "Current"])
drink = st.sidebar.radio("Drinking Habits", ['Never', 'Occasional', 'Regular', 'Heavy'])
exersice = st.sidebar.radio("Exersice", ['Sedentary', 'Light', 'Moderate', 'Intense'])

db_hist = int(st.sidebar.toggle("Family History of Diabetes"))
hd_hist = int(st.sidebar.toggle("Family History of Heart Disease"))
ob_hist = int(st.sidebar.toggle("Family History of Obesity"))

raw_data = {
    'age' :age,
    'gender' :gender,
    'bmi' : bmi,
    'systolic_bp' :s_bp,
    'diastolic_bp': d_bp,
    'smoking' : smoke,
    'drinking' : drink,
    'exercise':exersice,
    'stress_level':stress,
    'family_history_diabetes' : db_hist,
    'family_history_heart' : hd_hist,
    'family_history_obesity' : ob_hist
}
df = pd.DataFrame([raw_data])

if st.button("Analyze Health Risks", type="primary"):
    
    processed_input = predict.transform_input(df) 
    
    preds = my_model.predict(processed_input, verbose=0)
    
    levels = ["LOW", "MEDIUM", "HIGH"]
    
    st.divider()
    st.subheader("Diagnostic Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk = levels[np.argmax(preds[0])]
        st.metric(label="Diabetes Risk", value=risk)
        
    with col2:
        risk = levels[np.argmax(preds[1])]
        st.metric(label="Heart Disease Risk", value=risk)
        
    with col3:
        risk = levels[np.argmax(preds[2])]
        st.metric(label="Obesity Risk", value=risk)
        
    st.info("Note: This is an AI-generated screening tool and not a clinical diagnosis.")