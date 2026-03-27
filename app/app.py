# ============================================================
# E-Commerce Analytics & Recommendation App
# Built with Streamlit | Portfolio Project
# ============================================================ 

import streamlit as st
import pandas as pd 
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os

# ============================================================
# PAGE CONFIG — must be first Streamlit command
# ============================================================
st.set_page_config(
    page_title="E-Commerce Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — dark professional theme
# ============================================================
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0B0F1A; color: #E2E8F0; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color:#0B0F1A;
        border-right: 1px solid #1E2D4A;
    }

    /* KPI Cards */
    .kpi-card {
        background: #131929;
        border-radius: 12px;
        padding: 20px 24px;
        border-left: 4px solid;
        margin-bottom: 12px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        margin: 4px 0;
    }
    .kpi-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748B;
    }
    .kpi-blue   { border-color: #3B82F6; }
    .kpi-green  { border-color: #10B981; }
    .kpi-purple { border-color: #8B5CF6; }
    .kpi-amber  { border-color: #F59E0B; }

    /* Section headers */
    .section-title {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #64748B;
        margin-bottom: 16px;
        border-bottom: 1px solid #1E2D4A;
        padding-bottom: 8px;
    }

    /* Insight pills */
    .insight-pill {
        background: #1C2640;
        border-left: 3px solid #F59E0B;
        border-radius: 6px;
        padding: 10px 16px;
        font-size: 13px;
        color: #0B0F1A;
        margin-top: 8px;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer     {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA — cached so it only loads once
# ============================================================
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))

    df  = pd.read_csv(os.path.join(base, '../data/cleaned/cleaned_online_retail.csv'))
    rfm = pd.read_csv(os.path.join(base, '../data/rfm/rfm_segments.csv'))
    rec = pd.read_csv(os.path.join(base, '../data/recommendations/all_recommendations.csv'))

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)

    return df, rfm, rec

@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(base, '../data/churn model/churn_model.pkl'), 'rb') as f:
        model = pickle.load(f)

    with open(os.path.join(base, '../data/churn model/scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler
df, rfm, rec_df = load_data()
model, scaler   = load_model()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown("## 🛒 E-Commerce Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview",
     "👥 Customer Segments",
     "🔮 Churn Predictor",
     "🛍️ Product Recommender"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:11px; color:#FFFFFF;'>
UCI Online Retail Dataset<br>
397,884 transactions · 37 countries<br>
Dec 2010 – Dec 2011
</div>
""", unsafe_allow_html=True)

# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================
if page == "📊 Overview":

    st.markdown("# 📊 Sales Overview")
    st.markdown('<p class="section-title">Key Performance Indicators</p>',
                unsafe_allow_html=True)

    # KPI calculations
    total_revenue  = df['TotalPrice'].sum()
    total_orders   = df['InvoiceNo'].nunique()
    total_customers= df['CustomerID'].nunique()
    avg_order_val  = total_revenue / total_orders

    # KPI row
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="kpi-card kpi-blue">
            <div class="kpi-label">Total Revenue</div>
            <div class="kpi-value" style="color:#FFFFFF">
                £{total_revenue:,.0f}
            </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi-card kpi-green">
            <div class="kpi-label">Total Orders</div>
            <div class="kpi-value" style="color:#10B981">
                {total_orders:,}
            </div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi-card kpi-purple">
            <div class="kpi-label">Total Customers</div>
            <div class="kpi-value" style="color:#8B5CF6">
                {total_customers:,}
            </div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="kpi-card kpi-amber">
            <div class="kpi-label">Avg Order Value</div>
            <div class="kpi-value" style="color:#F59E0B">
                £{avg_order_val:,.0f}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Monthly Revenue Trend
    st.markdown('<p class="section-title">Monthly Revenue Trend</p>',
                unsafe_allow_html=True)

    monthly = (df.groupby('YearMonth')['TotalPrice']
               .sum().reset_index()
               .rename(columns={'TotalPrice':'Revenue'}))

    fig = px.area(
        monthly, x='YearMonth', y='Revenue',
        color_discrete_sequence=['#3B82F6'],
        template='plotly_dark'
    )
    fig.update_layout(
        paper_bgcolor='#131929', plot_bgcolor='#131929',
        font_color='#E2E8F0',
        xaxis_title='', yaxis_title='Revenue (£)',
        yaxis_tickprefix='£', yaxis_tickformat=',',
        margin=dict(l=20, r=20, t=20, b=20),
        height=300
    )
    fig.update_traces(line_color='#3B82F6', fillcolor='rgba(59,130,246,0.15)')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-pill">
    💡 <strong style="color:#F59E0B">Peak Insight:</strong>
    Revenue peaked in November 2011 — driven by pre-Christmas demand.
    Recommend increasing inventory and marketing spend in October each year.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Two charts side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-title">Top 10 Products by Revenue</p>',
                    unsafe_allow_html=True)
        top_prod = (df.groupby('Description')['TotalPrice']
                    .sum().sort_values(ascending=False)
                    .head(10).reset_index())
        top_prod.columns = ['Product', 'Revenue']
        top_prod['Product'] = top_prod['Product'].str[:30]

        fig2 = px.bar(
            top_prod, x='Revenue', y='Product',
            orientation='h',
            color_discrete_sequence=['#10B981'],
            template='plotly_dark'
        )
        fig2.update_layout(
            paper_bgcolor='#131929', plot_bgcolor='#131929',
            font_color='#E2E8F0', yaxis={'categoryorder':'total ascending'},
            xaxis_tickprefix='£', xaxis_tickformat=',',
            margin=dict(l=10, r=10, t=10, b=10), height=350
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<p class="section-title">Revenue by Country (Top 10)</p>',
                    unsafe_allow_html=True)
        country = (df.groupby('Country')['TotalPrice']
                   .sum().sort_values(ascending=False)
                   .head(10).reset_index())
        country.columns = ['Country', 'Revenue']

        fig3 = px.bar(
            country, x='Revenue', y='Country',
            orientation='h',
            color_discrete_sequence=['#8B5CF6'],
            template='plotly_dark'
        )
        fig3.update_layout(
            paper_bgcolor='#131929', plot_bgcolor='#131929',
            font_color='#E2E8F0', yaxis={'categoryorder':'total ascending'},
            xaxis_tickprefix='£', xaxis_tickformat=',',
            margin=dict(l=10, r=10, t=10, b=10), height=350
        )
        st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# PAGE 2 — CUSTOMER SEGMENTS
# ============================================================
elif page == "👥 Customer Segments":

    st.markdown("# 👥 Customer Segments")

    # Segment filter
    segments = ['All'] + sorted(rfm['Segment'].unique().tolist())
    selected = st.selectbox("Filter by Segment", segments)

    filtered = rfm if selected == 'All' else rfm[rfm['Segment'] == selected]

    # Metrics row
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers",      f"{len(filtered):,}")
    c2.metric("Avg Recency",    f"{filtered['Recency'].mean():.0f} days")
    c3.metric("Avg Spend",      f"£{filtered['Monetary'].mean():,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-title">Segment Distribution</p>',
                    unsafe_allow_html=True)

        seg_counts = rfm['Segment'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Count']

        color_map = {
            'Champions'       : '#10B981',
            'Loyal Customers' : '#3B82F6',
            'New Customers'   : '#6366F1',
            'Promising'       : '#8B5CF6',
            'At Risk'         : '#F59E0B',
            'Cannot Lose Them': '#EF4444',
            'Lost'            : '#6B7280',
            'Needs Attention' : '#F97316'
        }

        fig4 = px.pie(
            seg_counts, names='Segment', values='Count',
            color='Segment', color_discrete_map=color_map,
            hole=0.55, template='plotly_dark'
        )
        fig4.update_layout(
            paper_bgcolor='#131929',
            font_color='#E2E8F0',
            margin=dict(l=10, r=10, t=10, b=10),
            height=350,
            legend=dict(font=dict(size=11))
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col2:
        st.markdown('<p class="section-title">Recency vs Monetary</p>',
                    unsafe_allow_html=True)

        fig5 = px.scatter(
            rfm, x='Recency', y='Monetary',
            color='Segment', color_discrete_map=color_map,
            hover_data=['CustomerID', 'Frequency'],
            opacity=0.7, template='plotly_dark'
        )
        fig5.update_layout(
            paper_bgcolor='#131929', plot_bgcolor='#131929',
            font_color='#E2E8F0',
            yaxis_tickprefix='£', yaxis_tickformat=',',
            margin=dict(l=10, r=10, t=10, b=10),
            height=350
        )
        st.plotly_chart(fig5, use_container_width=True)

    # Segment table
    st.markdown("---")
    st.markdown('<p class="section-title">Segment Summary Table</p>',
                unsafe_allow_html=True)

    summary = rfm.groupby('Segment').agg(
        Customers     = ('CustomerID', 'count'),
        Avg_Recency   = ('Recency',    'mean'),
        Avg_Frequency = ('Frequency',  'mean'),
        Avg_Spend     = ('Monetary',   'mean'),
        Total_Revenue = ('Monetary',   'sum')
    ).round(1).reset_index().sort_values('Total_Revenue', ascending=False)

    st.dataframe(
        summary.style.format({
            'Avg_Spend'    : '£{:.0f}',
            'Total_Revenue': '£{:,.0f}'
        }),
        use_container_width=True,
        height=320
    )

# ============================================================
# PAGE 3 — CHURN PREDICTOR
# ============================================================
elif page == "🔮 Churn Predictor":

    st.markdown("# 🔮 Churn Probability Predictor")
    st.markdown("Enter customer details below to predict churn risk.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Enter Customer Details")

        recency   = st.slider("Recency (days since last purchase)",
                               1, 365, 60)
        frequency = st.slider("Frequency (number of orders)",
                               1, 100, 5)
        monetary  = st.number_input("Monetary (total spend £)",
                                     min_value=0.0, max_value=50000.0,
                                     value=500.0, step=50.0)
        r_score   = st.selectbox("Recency Score (1=worst, 4=best)",   [1,2,3,4], index=2)
        f_score   = st.selectbox("Frequency Score (1=worst, 4=best)", [1,2,3,4], index=2)
        m_score   = st.selectbox("Monetary Score (1=worst, 4=best)",  [1,2,3,4], index=2)
        total_score = r_score + f_score + m_score

        predict_btn = st.button("🔮 Predict Churn Risk", type="primary",
                                 use_container_width=True)

    with col2:
        st.markdown("### Prediction Result")

        if predict_btn:
            features = np.array([[recency, frequency, monetary,
                                   r_score, f_score, m_score, total_score]])

            prob  = model.predict_proba(features)[0][1]
            label = model.predict(features)[0]

            # Risk level
            if prob >= 0.7:
                risk_color = '#EF4444'
                risk_label = '🔴 High Risk'
                action     = 'Send retention offer immediately. Discount voucher or loyalty reward recommended.'
            elif prob >= 0.4:
                risk_color = '#F59E0B'
                risk_label = '🟡 Medium Risk'
                action     = 'Monitor closely. Send re-engagement email campaign within 2 weeks.'
            else:
                risk_color = '#10B981'
                risk_label = '🟢 Low Risk'
                action     = 'Customer is active. Continue standard engagement. Consider upsell opportunity.'

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = prob * 100,
                title = {'text': "Churn Probability", 'font': {'color': '#E2E8F0'}},
                number= {'suffix': '%', 'font': {'color': risk_color, 'size': 40}},
                gauge = {
                    'axis'      : {'range': [0, 100],
                                   'tickcolor': '#64748B'},
                    'bar'       : {'color': risk_color},
                    'bgcolor'   : '#1C2640',
                    'bordercolor': '#1E2D4A',
                    'steps'     : [
                        {'range': [0,  40], 'color': '#0D2137'},
                        {'range': [40, 70], 'color': '#1C2D1C'},
                        {'range': [70,100], 'color': '#2D1515'}
                    ],
                    'threshold' : {
                        'line' : {'color': '#F59E0B', 'width': 3},
                        'value': 70
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='#131929',
                font_color='#E2E8F0',
                height=280,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(f"""
            <div class="kpi-card" style="border-color:{risk_color}">
                <div class="kpi-label">Risk Level</div>
                <div class="kpi-value" style="color:{risk_color};font-size:22px">
                    {risk_label}
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="insight-pill">
            💡 <strong style="color:#F59E0B">Recommended Action:</strong><br>
            {action}
            </div>""", unsafe_allow_html=True)

        else:
            st.info("👈 Fill in the customer details and click Predict.")

# ============================================================
# PAGE 4 — PRODUCT RECOMMENDER
# ============================================================
elif page == "🛍️ Product Recommender":

    st.markdown("# 🛍️ Product Recommender")
    st.markdown("Enter a Customer ID to get personalised product recommendations.")
    st.markdown("---")

    # Customer ID input
    available_ids = sorted(rec_df['CustomerID'].unique().tolist())

    col1, col2 = st.columns([1, 2])

    with col1:
        customer_id = st.selectbox(
            "Select Customer ID",
            available_ids,
            index=0
        )
        n_recs = st.slider("Number of Recommendations", 3, 10, 5)
        rec_btn = st.button("🛍️ Get Recommendations",
                             type="primary", use_container_width=True)

    with col2:
        if rec_btn:
            # Customer profile
            if customer_id in rfm['CustomerID'].values:
                profile = rfm[rfm['CustomerID'] == customer_id].iloc[0]

                st.markdown("#### Customer Profile")
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Segment",   profile['Segment'])
                p2.metric("Recency",   f"{profile['Recency']} days")
                p3.metric("Orders",    int(profile['Frequency']))
                p4.metric("Spent",     f"£{profile['Monetary']:,.0f}")

            st.markdown("---")
            st.markdown("#### Recommended Products")

            recs = (rec_df[rec_df['CustomerID'] == customer_id]
                    .head(n_recs)
                    .reset_index(drop=True))

            if len(recs) == 0:
                st.warning("No recommendations found for this customer.")
            else:
                for _, row in recs.iterrows():
                    st.markdown(f"""
                    <div class="kpi-card kpi-purple" style="padding:14px 18px;margin-bottom:8px">
                        <div style="display:flex;justify-content:space-between;align-items:center">
                            <div>
                                <div class="kpi-label">Rank #{int(row['Rank'])}</div>
                                <div style="font-size:15px;font-weight:600;
                                            color:#E2E8F0;margin-top:4px">
                                    {row['Recommended']}
                                </div>
                            </div>
                            <div style="text-align:right">
                                <div class="kpi-label">Score</div>
                                <div style="font-size:18px;font-weight:700;
                                            color:#8B5CF6">
                                    {row['Score']:.0f}
                                </div>
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)
        else:
            st.info("👈 Select a Customer ID and click Get Recommendations.")
