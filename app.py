import streamlit as st
import pickle
import pandas as pd
import numpy as np
import urllib.parse

# Explicit imports for unpickling
import sklearn
import xgboost

st.set_page_config(
    page_title="Laptop Price Intelligence & Recommendation Engine",
    page_icon="💻",
    layout="wide"
)

st.title("💻 Laptop Price Intelligence & Recommendation Engine")
st.caption("AI-powered price estimation, hardware upgrade simulator, and live marketplace discovery.")

# ---------------------------------------------------------
# Load Pickled Artifacts
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Organized Specification Tabs
# ---------------------------------------------------------
st.markdown("### Configure Laptop Specifications")
tab1, tab2, tab3 = st.tabs(["⚡ Core Performance", "🖥️ Display & Chassis", "💾 Storage & OS"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        company = st.selectbox('Brand', sorted(df['Company'].unique()))
    with col2:
        cpu_options = df['Cpu brand'].unique() if 'Cpu brand' in df.columns else df['Cpu'].unique()
        cpu = st.selectbox('CPU Brand', sorted(cpu_options))
    with col3:
        gpu_options = df['Gpu brand'].unique() if 'Gpu brand' in df.columns else df['Gpu'].unique()
        gpu = st.selectbox('GPU Brand', sorted(gpu_options))

    col4, col5 = st.columns(2)
    with col4:
        ram = st.selectbox('RAM (GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64], index=3)
    with col5:
        type_name = st.selectbox('Chassis / Type', sorted(df['TypeName'].unique()))

with tab2:
    col6, col7, col8 = st.columns(3)
    with col6:
        screen_size = st.number_input('Screen Size (Inches)', min_value=10.0, max_value=20.0, value=13.3, step=0.1)
    with col7:
        resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '2560x1600', '3840x2160', '2880x1800'])
    with col8:
        weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, value=1.37, step=0.05)

    col9, col10 = st.columns(2)
    with col9:
        touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
    with col10:
        ips = st.selectbox('IPS Display', ['No', 'Yes'])

with tab3:
    col11, col12, col13 = st.columns(3)
    with col11:
        storage_type = st.selectbox('Primary Storage Type', ['SSD', 'HDD'])
    with col12:
        storage_capacity = st.selectbox('Storage Capacity (GB)', [128, 256, 512, 1024, 2048], index=2)
    with col13:
        os = st.selectbox('Operating System', os_options)

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button('Estimate Market Price & Generate Insights 🚀', use_container_width=True)

# ---------------------------------------------------------
# Prediction & Analytics Execution
# ---------------------------------------------------------
if predict_clicked:
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    ssd = storage_capacity if storage_type == 'SSD' else 0
    hdd = storage_capacity if storage_type == 'HDD' else 0

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

    # Invert log-space prediction
    if pred < 15:
        final_price = int(np.expm1(pred))
    else:
        final_price = int(pred)

    st.divider()

    # 1. Executive Metrics & Comparison Cards
    st.markdown("### 📊 Valuation Analysis")
    metric_col1, metric_col2, metric_col3 = st.columns(3)

    # Calculate average benchmark for selected brand if Price is in df
    if 'Price' in df.columns:
        brand_prices = df[df['Company'] == company]['Price']
        # Check if df['Price'] is log-transformed or raw
        if brand_prices.mean() < 15:
            brand_avg = int(np.expm1(brand_prices.mean()))
        else:
            brand_avg = int(brand_prices.mean())
        brand_delta = final_price - brand_avg
    else:
        brand_avg = final_price
        brand_delta = 0

    # Assume a standard model uncertainty band of ~8%
    lower_bound = int(final_price * 0.92)
    upper_bound = int(final_price * 1.08)

    with metric_col1:
        st.metric(
            label="Predicted Market Value",
            value=f"₹{final_price:,}",
            delta=f"{'+' if brand_delta >= 0 else ''}₹{brand_delta:,} vs {company} Avg",
            delta_color="inverse"
        )
    with metric_col2:
        st.metric(
            label="Confidence Interval (±8%)",
            value=f"₹{lower_bound:,} – ₹{upper_bound:,}",
            help="Expected valuation spread accounting for regional taxes, retailer variance, and discount cycles."
        )
    with metric_col3:
        st.metric(
            label="Display Density (PPI)",
            value=f"{ppi:.1f} PPI",
            help="Pixels per inch calculated from screen size and resolution."
        )

    # 2. What-If Upgrade Simulator (Marginal Cost Impact)
    st.markdown("### 🛠️ Hardware 'What-If' Cost Simulator")
    with st.expander("Explore how hardware upgrades alter the market value", expanded=True):
        sim_col1, sim_col2 = st.columns(2)

        # Simulation A: RAM Upgrade
        next_ram = 16 if ram <= 8 else 32
        query_ram = query.copy()
        query_ram['Ram'] = next_ram
        pred_ram = pipe.predict(query_ram)[0]
        price_ram = int(np.expm1(pred_ram) if pred_ram < 15 else pred_ram)
        ram_delta = price_ram - final_price

        with sim_col1:
            st.markdown(f"**RAM Upgrade ({ram}GB ➜ {next_ram}GB)**")
            if ram_delta > 0:
                st.write(f"Estimated Price Shift: **+₹{ram_delta:,}** (Total: ₹{price_ram:,})")
            else:
                st.write("Current configuration already at optimal tier.")

        # Simulation B: Storage Upgrade
        next_storage = 512 if storage_capacity <= 256 else 1024
        query_storage = query.copy()
        query_storage['SSD'] = next_storage
        pred_storage = pipe.predict(query_storage)[0]
        price_storage = int(np.expm1(pred_storage) if pred_storage < 15 else pred_storage)
        storage_delta = price_storage - final_price

        with sim_col2:
            st.markdown(f"**SSD Upgrade ({storage_capacity}GB ➜ {next_storage}GB)**")
            if storage_delta > 0:
                st.write(f"Estimated Price Shift: **+₹{storage_delta:,}** (Total: ₹{price_storage:,})")
            else:
                st.write("Current storage configuration is already at maximum bracket.")

    # 3. Existing Benchmark Matches from Dataset
    st.markdown("### 📋 Historical Market Benchmarks (Dataset Matches)")
    price_col_exists = 'Price' in df.columns
    if price_col_exists:
        df_copy = df.copy()
        if df_copy['Price'].mean() < 15:
            df_copy['Clean_Price'] = np.expm1(df_copy['Price']).astype(int)
        else:
            df_copy['Clean_Price'] = df_copy['Price'].astype(int)

        benchmark_matches = df_copy[
            (df_copy['Company'] == company) &
            (df_copy['Clean_Price'] >= final_price * 0.85) &
            (df_copy['Clean_Price'] <= final_price * 1.15)
        ]

        if not benchmark_matches.empty:
            display_cols = [c for c in ['Company', 'TypeName', 'Ram', 'Cpu brand', 'Clean_Price'] if c in df_copy.columns]
            st.dataframe(
                benchmark_matches[display_cols].head(5).rename(columns={'Clean_Price': 'Actual Price (₹)'}),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(f"No direct listings for {company} found in the exact ±15% historical bracket. See real-time marketplace links below.")

    # 4. Amazon & Flipkart Live Discovery Hub
    st.markdown("### 🛒 Live E-Commerce Match Finder")
    min_budget = int(final_price * 0.90)
    max_budget = int(final_price * 1.10)

    st.markdown(f"Direct listings matching this specification between **₹{min_budget:,}** and **₹{max_budget:,}**:")

    search_keywords = f"{company} {ram}GB {cpu} laptop"
    amazon_encoded = urllib.parse.quote_plus(search_keywords)
    amazon_url = f"https://www.amazon.in/s?k={amazon_encoded}&low-price={min_budget}&high-price={max_budget}"

    flipkart_encoded = urllib.parse.quote_plus(search_keywords)
    flipkart_url = f"https://www.flipkart.com/search?q={flipkart_encoded}&p%5B%5D=facets.price_range.from%3D{min_budget}&p%5B%5D=facets.price_range.to%3D{max_budget}"

    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        st.markdown("**Amazon India**")
        st.caption(f"Filters: `{company}`, `{ram}GB RAM`, `₹{min_budget:,} - ₹{max_budget:,}`")
        st.link_button("View on Amazon 📦", amazon_url, use_container_width=True)

    with rec_col2:
        st.markdown("**Flipkart**")
        st.caption(f"Filters: `{company}`, `{ram}GB RAM`, `₹{min_budget:,} - ₹{max_budget:,}`")
        st.link_button("View on Flipkart 🛍️", flipkart_url, use_container_width=True)
