# AI-Powered Health Policy Impact Simulator (India Focus)

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Simulated expanded dataset for all Indian states
states = [
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat',
    'Haryana', 'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra',
    'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim',
    'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
]

np.random.seed(42)
data = {
    'State': states,
    'Doctors_per_1000': np.random.uniform(0.4, 1.8, len(states)),
    'Clinic_Coverage_%': np.random.uniform(30, 95, len(states)),
    'Telemedicine_%': np.random.uniform(10, 80, len(states)),
    'Funding_Cr': np.random.uniform(100, 1000, len(states)),
    'Literacy_Rate_%': np.random.uniform(60, 95, len(states)),
    'Rural_Population_%': np.random.uniform(40, 85, len(states)),
    'Sanitation_Access_%': np.random.uniform(30, 95, len(states)),
    'Internet_Penetration_%': np.random.uniform(20, 90, len(states)),
    'Infant_Mortality_Rate': np.random.uniform(18, 55, len(states)),
    'Maternal_Mortality_Rate': np.random.uniform(50, 200, len(states)),
    'Life_Expectancy': np.random.uniform(62, 75, len(states))
}

df = pd.DataFrame(data)

# Train ML models for 3 health KPIs
features = ['Doctors_per_1000', 'Clinic_Coverage_%', 'Telemedicine_%', 'Funding_Cr',
            'Literacy_Rate_%', 'Rural_Population_%', 'Sanitation_Access_%', 'Internet_Penetration_%']

X = df[features]

# Train models
models = {}
predictions = {}
for target in ['Infant_Mortality_Rate', 'Maternal_Mortality_Rate', 'Life_Expectancy']:
    y = df[target]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    models[target] = model
    predictions[target] = model.predict(X)

# Streamlit UI
st.set_page_config(layout="wide")
st.title("🇮🇳 India HealthSim: AI-Driven Health Policy Simulator")

st.sidebar.header("🛠 Policy Inputs")
state_selected = st.sidebar.selectbox("Select State", df['State'])

input_values = {
    'Doctors_per_1000': st.sidebar.slider("Doctors per 1000", 0.3, 2.0, 1.0, 0.1),
    'Clinic_Coverage_%': st.sidebar.slider("Clinic Coverage %", 10, 100, 60),
    'Telemedicine_%': st.sidebar.slider("Telemedicine Access %", 5, 100, 30),
    'Funding_Cr': st.sidebar.slider("Health Funding (Cr)", 100, 2000, 500, 50),
    'Literacy_Rate_%': st.sidebar.slider("Literacy Rate %", 50, 100, 75),
    'Rural_Population_%': st.sidebar.slider("Rural Population %", 30, 90, 65),
    'Sanitation_Access_%': st.sidebar.slider("Sanitation Access %", 10, 100, 70),
    'Internet_Penetration_%': st.sidebar.slider("Internet Penetration %", 10, 100, 60)
}

input_df = pd.DataFrame([input_values])

# Display projections
st.subheader(f"📊 Projected Health Outcomes for {state_selected}")

cols = st.columns(3)
metrics = ['Infant_Mortality_Rate', 'Maternal_Mortality_Rate', 'Life_Expectancy']
units = ['deaths/1000 births', 'deaths/100,000 births', 'years']
for i, metric in enumerate(metrics):
    prediction = models[metric].predict(input_df)[0]
    current_value = df[df['State'] == state_selected][metric].values[0]
    delta = current_value - prediction if metric != 'Life_Expectancy' else prediction - current_value
    with cols[i]:
        st.metric(label=metric.replace("_", " "), value=f"{prediction:.2f} {units[i]}", delta=f"{delta:.2f}")

# Comparative chart
st.subheader("📈 Current vs Projected (All Metrics)")
comparison_df = pd.DataFrame({
    'Health Metric': metrics,
    'Current': [df[df['State'] == state_selected][m].values[0] for m in metrics],
    'Projected': [models[m].predict(input_df)[0] for m in metrics]
})
comparison_df.set_index("Health Metric", inplace=True)
st.bar_chart(comparison_df)
