import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Explicit imports for unpickling
import sklearn
import xgboost

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

st.title("💻 Laptop Price Predictor")
st.write("Enter the hardware specifications below to predict the estimated market price.")

# Load pickled models safely
@st.cache_resource
def load_artifacts():
    df = pickle.load(open('df.pkl(viru)', 'rb'))
    pipe = pickle.load(open('pipe.pkl(viru)', 'rb'))
    return df, pipe

df, pipe = load_artifacts()

# Helper function to find the exact OS column name present in df
def get_os_column(df):
    possible_names = ['OpSys', 'os', 'OS', 'OpSys_brand', 'Op_Sys']
    for col in possible_names:
        if col in df.columns:
            return col
    return None

os_col = get_os_column(df)
os_options = df[os_col].unique() if os_col else ['Windows', 'macOS', 'Linux', 'Other']

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('Brand', df['Company'].unique())
    type_name = st.selectbox('Type', df['TypeName'].unique())
    ram = st.selectbox('RAM (GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64], index=3)
    weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, value=1.37, step=0.1)
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
    ips = st.selectbox('IPS Display', ['No', 'Yes'])

with col2:
    screen_size = st.number_input('Screen Size (Inches)', min_value=10.0, max_value=20.0, value=13.3, step=0.1)
    resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '2560x1600', '3840x2160', '2880x1800'])
    cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique() if 'Cpu brand' in df.columns else df['Cpu'].unique())
    gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique() if 'Gpu brand' in df.columns else df['Gpu'].unique())
    os = st.selectbox('Operating System', os_options)
    storage_type = st.selectbox('Primary Storage', ['SSD', 'HDD'])
    storage_capacity = st.selectbox('Storage Capacity (GB)', [128, 256, 512, 1024])

if st.button('Predict Price 🚀', use_container_width=True):
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0
    
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size
    
    ssd = storage_capacity if storage_type == 'SSD' else 0
    hdd = storage_capacity if storage_type == 'HDD' else 0

    # Match OS column key with what pipe expects
    os_feature_key = os_col if os_col else 'OpSys'

    query = pd.DataFrame([{
        'Company': company,
        'TypeName': type_name,
        'Ram': ram,
        'Weight': weight,
        'Touchscreen': touchscreen_val,
        'Ips': ips_val,
        'ppi': ppi,
        'Cpu brand': cpu,
        'HDD': hdd,
        'SSD': ssd,
        'Gpu brand': gpu,
        os_feature_key: os
    }])
    
    pred = pipe.predict(query)[0]
    
    # Reverse log-transform if y = np.log(Price) was used during training
    if pred < 15:
        pred = np.expm1(pred)
        
    st.success(f"### Estimated Price: ₹ {int(pred):,}")
