import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

st.title("💻 Laptop Price Predictor")
st.write("Enter the hardware specifications below to predict the estimated market price.")

# Load pickled models with exact file names matching your GitHub repo
@st.cache_resource
def load_artifacts():
    # Exact names as shown in your GitHub file tree
    df = pickle.load(open('df.pkl(viru)', 'rb'))
    pipe = pickle.load(open('pipe.pkl(viru)', 'rb'))
    return df, pipe

df, pipe = load_artifacts()

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
    cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique())
    gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())
    os = st.selectbox('Operating System', df['OpSys'].unique())
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
        'OpSys': os
    }])
    
    pred = pipe.predict(query)[0]
    
    # Reverse log transform if y = np.log(Price) was used during training
    if pred < 15:
        pred = np.expm1(pred)
        
    st.success(f"### Estimated Price: ₹ {int(pred):,}")
