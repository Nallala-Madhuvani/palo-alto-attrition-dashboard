import streamlit as st
import pandas as pd
import plotly.express as px

# Page layout configuration
st.set_page_config(page_title="Palo Alto Networks - Workforce Attrition", layout="wide")
st.title("🛡️ Palo Alto Networks: Workforce Attrition & Risk Hotspot Analysis")

# 1. Load and prepare dataset
@st.cache_data
def load_and_preprocess():
    data = pd.read_csv('Palo Alto Networks.csv')
    
    # Standardize Attrition to numeric (1/0) and label
    if data['Attrition'].dtype == 'object':
        data['Attrition_Num'] = data['Attrition'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
    else:
        data['Attrition_Num'] = data['Attrition'].astype(int)
    
    data['Attrition_Label'] = data['Attrition_Num'].map({1: 'Exited', 0: 'Retained'})

    # Feature Engineering: Age & Tenure Buckets
    data['AgeGroup'] = pd.cut(
        data['Age'], 
        bins=[17, 25, 35, 45, 55, 100], 
        labels=['<25', '25-34', '35-44', '45-54', '55+']
    )
    
    data['TenureBucket'] = pd.cut(
        data['YearsAtCompany'], 
        bins=[-1, 2, 5, 10, 45], 
        labels=['0-2 yrs (Early)', '3-5 yrs (Mid)', '6-10 yrs', '10+ yrs (Senior)']
    )
    
    return data

df = load_and_preprocess()

# 2. Sidebar Filters
st.sidebar.header("🔍 Filter Workforce Data")

selected_dept = st.sidebar.multiselect(
    "Department", 
    options=df['Department'].unique().tolist(), 
    default=df['Department'].unique().tolist()
)

available_roles = df[df['Department'].isin(selected_dept)]['JobRole'].unique().tolist()
selected_roles = st.sidebar.multiselect(
    "Job Role", 
    options=available_roles, 
    default=available_roles
)

tenure_min, tenure_max = int(df['YearsAtCompany'].min()), int(df['YearsAtCompany'].max())
selected_tenure = st.sidebar.slider(
    "Tenure at Company (Years)", 
    min_value=tenure_min, 
    max_value=tenure_max, 
    value=(tenure_min, tenure_max)
)

ot_filter = st.sidebar.radio("OverTime Workload", options=["All", "Yes", "No"], index=0)

# Filter applied data
filtered_df = df[
    (df['Department'].isin(selected_dept)) &
    (df['JobRole'].isin(selected_roles)) &
    (df['YearsAtCompany'].between(selected_tenure[0], selected_tenure[1]))
]
if ot_filter != "All":
    filtered_df = filtered_df[filtered_df['OverTime'] == ot_filter]

# Check if data exists after filtering
if filtered_df.empty:
    st.warning("No records match the selected filters. Please expand your filter selections.")
    st.stop()

# 3. Main Dashboard Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Attrition Overview", 
    "🏢 Department & Role Hotspots", 
    "👥 Demographic Explorer", 
    "⚡ Workload & Mobility Impact"
])

# --- TAB 1: OVERVIEW ---
with tab1:
    total_emp = len(filtered_df)
    exited_emp = filtered_df['Attrition_Num'].sum()
    retained_emp = total_emp - exited_emp
    attr_rate = (exited_emp / total_emp) * 100 if total_emp > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Headcount", f"{total_emp:,}")
    col2.metric("Retained Employees", f"{retained_emp:,}")
    col3.metric("Departures", f"{exited_emp:,}")
    col4.metric("Attrition Rate", f"{attr_rate:.1f}%")

    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c1:
        fig_donut = px.pie(
            filtered_df, 
            names='Attrition_Label', 
            title="Retention vs. Departure Distribution",
            hole=0.45,
            color='Attrition_Label',
            color_discrete_map={'Retained': '#2ca02c', 'Exited': '#d62728'}
        )
        st.plotly_chart(fig_donut, use_container_width=True)
    with c2:
        tenure_dist = filtered_df.groupby('TenureBucket', observed=False)['Attrition_Num'].mean().reset_index()
        tenure_dist['Attrition Rate (%)'] = tenure_dist['Attrition_Num'] * 100
        fig_tenure = px.bar(
            tenure_dist, 
            x='TenureBucket', 
            y='Attrition Rate (%)',
            title="Attrition Rate by Tenure Stage",
            text_auto='.1f',
            color='Attrition Rate (%)',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_tenure, use_container_width=True)

# --- TAB 2: DEPARTMENT & ROLE HOTSPOTS ---
with tab2:
    st.subheader("Turnover Concentration by Function")
    col_dept, col_role = st.columns(2)
    
    with col_dept:
        dept_summary = filtered_df.groupby('Department')['Attrition_Num'].agg(
            Total='count', 
            Exits='sum', 
            Rate=lambda x: (x.sum() / x.count()) * 100
        ).reset_index()
        fig_dept = px.bar(
            dept_summary, 
            x='Department', 
            y='Rate', 
            title="Attrition Rate by Department (%)",
            text_auto='.1f',
            color='Rate',
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig_dept, use_container_width=True)

    with col_role:
        role_summary = filtered_df.groupby('JobRole')['Attrition_Num'].agg(
            Total='count', 
            Exits='sum', 
            Rate=lambda x: (x.sum() / x.count()) * 100
        ).reset_index().sort_values('Rate', ascending=True)
        fig_role = px.bar(
            role_summary, 
            x='Rate', 
            y='JobRole', 
            orientation='h',
            title="Attrition Rate by Job Role (%)",
            text_auto='.1f',
            color='Rate',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_role, use_container_width=True)

    st.markdown("### Department vs. Job Role Matrix")
    matrix = pd.crosstab(
        filtered_df['JobRole'], 
        filtered_df['Department'], 
        values=filtered_df['Attrition_Num'], 
        aggfunc='mean'
    ).fillna(0) * 100
    fig_heat = px.imshow(matrix, text_auto='.1f', aspect='auto', color_continuous_scale='Reds', title="Attrition Heatmap (%)")
    st.plotly_chart(fig_heat, use_container_width=True)

# --- TAB 3: DEMOGRAPHICS ---
with tab3:
    col_demo1, col_demo2 = st.columns(2)
    with col_demo1:
        age_summary = filtered_df.groupby('AgeGroup', observed=False)['Attrition_Num'].mean().reset_index()
        age_summary['Rate (%)'] = age_summary['Attrition_Num'] * 100
        fig_age = px.bar(age_summary, x='AgeGroup', y='Rate (%)', text_auto='.1f', title="Attrition Rate by Age Group (%)", color='Rate (%)', color_continuous_scale='Blues')
        st.plotly_chart(fig_age, use_container_width=True)

    with col_demo2:
        marital_summary = filtered_df.groupby('MaritalStatus')['Attrition_Num'].mean().reset_index()
        marital_summary['Rate (%)'] = marital_summary['Attrition_Num'] * 100
        fig_marital = px.bar(marital_summary, x='MaritalStatus', y='Rate (%)', text_auto='.1f', title="Attrition Rate by Marital Status (%)", color='Rate (%)', color_continuous_scale='Purples')
        st.plotly_chart(fig_marital, use_container_width=True)

    st.subheader("Compensation Disparity Check")
    fig_income = px.box(
        filtered_df, 
        x='JobLevel', 
        y='MonthlyIncome', 
        color='Attrition_Label',
        title="Monthly Income by Job Level vs. Attrition Status",
        color_discrete_map={'Retained': '#1f77b4', 'Exited': '#ff7f0e'}
    )
    st.plotly_chart(fig_income, use_container_width=True)

# --- TAB 4: WORKLOAD & MOBILITY ---
with tab4:
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        ot_summary = filtered_df.groupby(['OverTime', 'BusinessTravel'], observed=False)['Attrition_Num'].mean().reset_index()
        ot_summary['Rate (%)'] = ot_summary['Attrition_Num'] * 100
        fig_ot = px.bar(
            ot_summary, 
            x='BusinessTravel', 
            y='Rate (%)', 
            color='OverTime', 
            barmode='group',
            title="Impact of OverTime & Travel Frequency on Attrition (%)",
            text_auto='.1f'
        )
        st.plotly_chart(fig_ot, use_container_width=True)

    with col_w2:
        fig_commute = px.histogram(
            filtered_df, 
            x='DistanceFromHome', 
            color='Attrition_Label', 
            barmode='overlay',
            nbins=20,
            title="Commute Distance Distribution (Home to Office)",
            color_discrete_map={'Retained': '#2ca02c', 'Exited': '#d62728'}
        )
        st.plotly_chart(fig_commute, use_container_width=True)