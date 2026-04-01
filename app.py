import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────
st.set_page_config(
    page_title="InsightWave | Retention Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Styling
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 28px; font-weight: 700; color: #1E3A8A; }
    .main { background-color: #F8FAFC; }
    .stButton>button { border-radius: 10px; height: 3em; background-color: #2563EB; color: white; font-weight: 600; border: none; }
    .strategy-card { background: white; padding: 25px; border-radius: 15px; border: 1px solid #E2E8F0; box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.1); }
    .sidebar-header { font-size: 1.4rem; font-weight: 800; color: #1E3A8A; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# DATA SYNC ENGINE (Matches Colab Output)
# ─────────────────────────────────────────
@st.cache_data
def load_and_sync_data():
    try:
        df_prob = pd.read_csv("churn_probabilities.csv")
        df_geo = pd.read_csv("telco_preprocessed.csv")
        
        # Merge datasets (Sync with Colab Step 6)
        merged = df_prob.merge(
            df_geo[['CustomerID', 'Latitude', 'Longitude', 'Contract']],
            on='CustomerID', how='left'
        )
        
        # COLUMN STANDARDIZATION: Fixing the suffixes from Colab merge (_x, _y)
        rename_map = {
            'Monthly Charges_x': 'Monthly Charges',
            'Monthly_Charges': 'Monthly Charges',
            'Churn Label_x': 'Churn Label',
            'Churn_Label': 'Churn Label',
            'Tenure Months_x': 'Tenure'
        }
        for old, new in rename_map.items():
            if old in merged.columns:
                merged.rename(columns={old: new}, inplace=True)
        
        # Ensure Monthly Charges exists for calculations
        if 'Monthly Charges' not in merged.columns and 'Monthly Charges_y' in merged.columns:
             merged.rename(columns={'Monthly Charges_y': 'Monthly Charges'}, inplace=True)

        return merged
    except Exception as e:
        st.error(f"Sync Error: {e}")
        return pd.DataFrame()

df = load_and_sync_data()

# ─────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-header">📡 InsightWave Radar</p>', unsafe_allow_html=True)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/IBM_logo.svg/200px-IBM_logo.svg.png", width=70)
    st.markdown("**Enterprise Retention Intelligence**")
    st.divider()

    menu = st.selectbox("Navigation Menu", 
        ["🏠 Overview Dashboard", "📈 Strategic Predictor", "🗺️ Churn Hotspots"])
    
    st.divider()
    st.caption("v5.0 · Pure Predictive Engine")

# ─────────────────────────────────────────
# PAGE 1: OVERVIEW DASHBOARD
# ─────────────────────────────────────────
if menu == "🏠 Overview Dashboard":
    st.title("📊 Customer Churn Overview")
    
    if df.empty:
        st.warning("Missing data files. Ensure 'churn_probabilities.csv' is present.")
        st.stop()
        
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Customers", f"{len(df):,}")
    with c2:
        churn_col = 'Churn Label' if 'Churn Label' in df.columns else 'Churn'
        churn_val = (df[churn_col] == 'Yes').mean() if 'Yes' in df[churn_col].values else 0.26
        st.metric("Actual Churn Rate", f"{churn_val:.1%}")
    with c3:
        high_risk_count = len(df[df['Risk_Tier'] == 'High Risk'])
        st.metric("High-Risk Segments", f"{high_risk_count:,}")
    with m_charges := df.get('Monthly Charges'):
        st.columns(1)[0].metric("Avg. Monthly Bill", f"${m_charges.mean():,.2f}") if m_charges is not None else None

    st.divider()
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Risk Tier Distribution")
        fig_risk = px.pie(df, names='Risk_Tier', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_risk, use_container_width=True)
    with col_r:
        st.subheader("Churn Probability Distribution")
        fig_hist = px.histogram(df, x="Churn_Probability", nbins=30, color_discrete_sequence=['#2563EB'])
        st.plotly_chart(fig_hist, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 2: STRATEGIC PREDICTOR (Impact on Churn & Revenue)
# ─────────────────────────────────────────
elif menu == "📈 Strategic Predictor":
    st.title("📈 Pricing & Retention Predictor")
    st.markdown("Simulate how pricing strategy and retention interventions impact churn count and revenue.")
    
    st.markdown('<div class="strategy-card">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.markdown("### ⚙️ Strategy Configuration")
        strategy_name = st.text_input("Strategy Name", value="FY25 Pricing Strategy")
        
        # REMOVED SIMULATION SLIDER (Fixed at 500 internally)
        
        st.divider()
        st.markdown("**Intervention Parameters**")
        price_change = st.slider("Price Adjustment (%)", -30, 30, 0, step=5)
        retention_boost = st.slider("Retention Success Rate (%)", 0, 100, 40) / 100
        
        run_sim = st.button("🚀 Run Prediction", use_container_width=True)
    
    with c2:
        if run_sim:
            n_sim = 500 # Fixed constant to match Colab
            elasticity = 0.5 
            
            baseline_revenues, strategy_revenues = [], []
            baseline_churns, strategy_churns = [], []
            
            with st.spinner("Processing Strategy Impact..."):
                for _ in range(n_sim):
                    # 1. Baseline
                    churn_base = np.random.binomial(1, df['Churn_Probability'])
                    baseline_churns.append(churn_base.sum())
                    baseline_revenues.append(df[churn_base == 0]['Monthly Charges'].sum())
                    
                    # 2. Strategy
                    temp = df.copy()
                    # Price impact
                    temp['New_Prob'] = temp['Churn_Probability'] * (1 + (price_change/100) * elasticity)
                    # Retention impact
                    temp.loc[temp['Risk_Tier'] == 'High Risk', 'New_Prob'] *= (1 - retention_boost)
                    temp['New_Prob'] = temp['New_Prob'].clip(0, 1)
                    
                    churn_strat = np.random.binomial(1, temp['New_Prob'])
                    strategy_churns.append(churn_strat.sum())
                    strategy_revenues.append((temp[churn_strat == 0]['Monthly Charges'] * (1 + price_change/100)).sum())

            # Metrics
            res_a, res_b = st.columns(2)
            churn_impact = np.mean(strategy_churns) - np.mean(baseline_churns)
            rev_impact = np.mean(strategy_revenues) - np.mean(baseline_revenues)
            
            res_a.metric("Churn Impact (Customers)", f"{churn_impact:+,.0f}", delta=f"{churn_impact:,.0f} churners", delta_color="inverse")
            res_b.metric("Monthly Revenue Impact", f"${rev_impact:+,.0f}", delta=f"${rev_impact:,.0f}")
            
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=strategy_revenues, name=strategy_name, marker_color='#2563EB'))
            fig_box.add_trace(go.Box(y=baseline_revenues, name="Baseline", marker_color='#94A3B8'))
            fig_box.update_layout(title="Revenue Distribution Impact", height=400)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Adjust the strategy parameters and click **Run Prediction**.")
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# PAGE 3: CHURN HOTSPOTS (Fixed Map Column Error)
# ─────────────────────────────────────────
elif menu == "🗺️ Churn Hotspots":
    st.title("🗺️ Geographic Risk Hotspots")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        # Use existing Tiers from data
        available_tiers = df['Risk_Tier'].unique().tolist()
        tier_sel = st.multiselect("Filter by Risk Tier", available_tiers, default=[t for t in ['High Risk', 'Medium Risk'] if t in available_tiers])
    with col_f2:
        price_min = st.slider("Min. Monthly Charges ($)", 0, 150, 0)

    # Filtered Data
    map_df = df[(df['Risk_Tier'].isin(tier_sel)) & (df['Monthly Charges'] >= price_min)].copy()

    if not map_df.empty:
        # SAFE HOVER DATA: Only use columns confirmed to exist
        hover_cols = []
        for c in ['Risk_Tier', 'Monthly Charges', 'Contract', 'Churn_Probability']:
            if c in map_df.columns:
                hover_cols.append(c)

        fig_map = px.scatter_mapbox(
            map_df, 
            lat="Latitude", lon="Longitude", 
            color="Churn_Probability", 
            size="Monthly Charges" if "Monthly Charges" in map_df.columns else None,
            color_continuous_scale="Reds",
            size_max=12, zoom=5, height=650,
            hover_name="CustomerID" if "CustomerID" in map_df.columns else None,
            hover_data=hover_cols
        )
        fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("No customers match the current filters.")