import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------- Page setup ----------------------------
st.set_page_config(
    page_title="HomeValue | Linear Regression",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#17233b; --muted:#6b7890; --blue:#3977e8; --pale:#f3f7ff; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(180deg,#f7faff 0%,#ffffff 55%,#f7f9fd 100%); color:var(--ink); }
.block-container { padding-top:1.6rem; padding-bottom:3rem; max-width:1440px; }
[data-testid="stSidebar"] { background:#f1f5fc; border-right:1px solid #e3eaf5; }
.hero {
  padding:30px 34px; border-radius:24px; color:#fff;
  background:linear-gradient(115deg,rgba(21,50,99,.94),rgba(54,112,206,.84)),
  url('https://images.unsplash.com/photo-1600585154340-be6161a56a0c?auto=format&fit=crop&w=1800&q=85') center/cover;
  box-shadow:0 16px 38px rgba(37,82,153,.16); margin-bottom:22px;
}
.hero h1 {font-family:Manrope,sans-serif;font-size:2.35rem;line-height:1.15;margin:0 0 9px 0;color:white;}
.hero p {font-size:1rem;color:#e8f0ff;margin:0;max-width:720px;}
.eyebrow {text-transform:uppercase;letter-spacing:.15em;font-size:.72rem;font-weight:800;color:#d4e4ff;margin-bottom:12px;}
.metric-card {
  background:#fff;border:1px solid #e7edf6;border-radius:18px;padding:18px 20px;
  box-shadow:0 5px 18px rgba(30,55,90,.045); transition:all .22s ease; min-height:112px;
}
.metric-card:hover {transform:translateY(-5px);box-shadow:0 14px 28px rgba(45,91,160,.13);border-color:#b9d0fa;}
.metric-label {font-size:.82rem;color:#71809a;font-weight:600;margin-bottom:8px;}
.metric-value {font-family:Manrope,sans-serif;font-size:1.65rem;font-weight:800;color:#1d3153;}
.metric-note {font-size:.75rem;color:#8a97aa;margin-top:4px;}
.section-title {font-family:Manrope,sans-serif;font-size:1.25rem;font-weight:800;color:#1b2e4d;margin:14px 0 4px;}
.section-sub {color:#77849a;font-size:.88rem;margin-bottom:12px;}
.info-tile {
 background:#fff;border:1px solid #e7edf6;border-radius:18px;padding:18px;height:100%;
 transition:transform .2s ease,box-shadow .2s ease;
}
.info-tile:hover {transform:translateY(-4px);box-shadow:0 12px 25px rgba(45,91,160,.10);}
.info-icon {font-size:1.7rem;margin-bottom:8px;}
div.stButton>button {border-radius:12px;border:0;background:linear-gradient(90deg,#3977e8,#6a8ff0);color:white;font-weight:700;padding:.62rem 1.2rem;transition:all .2s ease;}
div.stButton>button:hover {transform:translateY(-2px);box-shadow:0 8px 18px rgba(57,119,232,.25);color:white;}
div[data-testid="stTabs"] button {font-weight:700;}
.footer {text-align:center;color:#98a4b5;font-size:.78rem;padding:28px 0 8px;}
div[data-testid="stDataFrame"] {border:1px solid #e7edf6;border-radius:14px;overflow:hidden;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] label,
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stCaptionContainer"] { color:#17233b !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------- Data and model ----------------------------
DEFAULT_DATA = Path(__file__).parent / "dataset" / "Housing.csv"

@st.cache_data
def load_csv(file_or_path):
    return pd.read_csv(file_or_path)

@st.cache_resource
def train_regression(data, target, feature_names, test_size, seed):
    clean = data[feature_names + [target]].dropna().copy()
    X = clean[feature_names]
    y = clean[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, X_train, X_test, y_train, y_test, predictions, clean

st.sidebar.markdown("## 🏡 HomeValue")
st.sidebar.caption("Linear Regression Studio")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Explore Dataset", "Model Performance", "Price Prediction"],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.caption("House Price Estimation with Linear Regression")
st.sidebar.caption("Interactive Housing Market Analytics")

# Load selected dataset
uploaded = st.sidebar.file_uploader("Upload your own CSV (optional)", type=["csv"])
try:
    data = load_csv(uploaded) if uploaded is not None else load_csv(DEFAULT_DATA)
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

if data.empty:
    st.error("The dataset is empty.")
    st.stop()

# Friendly default target; allow user to choose numeric target
numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
if len(numeric_columns) < 2:
    st.error("Please provide a CSV with at least one numeric feature and one numeric target.")
    st.stop()

target_default = "price" if "price" in numeric_columns else numeric_columns[-1]
target = st.sidebar.selectbox("Target column", numeric_columns, index=numeric_columns.index(target_default))
feature_options = [c for c in numeric_columns if c != target]
features = st.sidebar.multiselect(
    "Input features",
    feature_options,
    default=feature_options[:min(3, len(feature_options))]
)
test_fraction = st.sidebar.slider("Test data (%)", 10, 40, 20, 5) / 100
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

if not features:
    st.warning("Select at least one input feature in the sidebar.")
    st.stop()

clean = data[features + [target]].dropna()
if len(clean) < 5:
    st.error("At least five complete rows are required to train the model.")
    st.stop()

model, X_train, X_test, y_train, y_test, predictions, clean = train_regression(
    data, target, features, test_fraction, int(seed)
)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

def metric_card(label, value, note):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True
    )

def section(title, subtitle=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)

# ---------------------------- Pages ----------------------------
if page == "Overview":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">House Price Estimation </div>
      <h1>Understand the value<br>behind every home.</h1>
      <p>Explore the Data. Understand the Trends. Predict the Price.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### MY project at a glance")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("DATA RECORDS", f"{len(data):,}", "Rows in the loaded dataset")
    with c2: metric_card("MODEL FEATURES", f"{len(features)}", "Selected numeric inputs")
    with c3: metric_card("R² SCORE", f"{r2:.3f}", "Explained variance on test data")
    with c4: metric_card("TEST MAE", f"{mae:,.0f}", "Average absolute error")

    st.markdown("")
    section("Explore the workspace", "Choose a section from the sidebar to investigate each part of the regression workflow.")
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="info-tile"><div class="info-icon">📊</div><b>Explore the data</b><p style="color:#71809a">Review sample records, distributions, and feature relationships.</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="info-tile"><div class="info-icon">🧠</div><b>Evaluate the model</b><p style="color:#71809a">Compare actual and predicted values and inspect regression coefficients.</p></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="info-tile"><div class="info-icon">🏠</div><b>Predict a price</b><p style="color:#71809a">Enter property details and get an estimate from the trained model.</p></div>', unsafe_allow_html=True)

    st.markdown("")
    section("Dataset preview")
    st.dataframe(data.head(8), use_container_width=True, hide_index=True)

elif page == "Explore Dataset":
    st.markdown('<div class="hero"><div class="eyebrow">DATA EXPLORATION</div><h1>Know your data.</h1><p>Inspect the dataset, compare feature distributions, and discover relationships before modeling.</p></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("ROWS", f"{data.shape[0]:,}", "Total observations")
    with c2: metric_card("COLUMNS", f"{data.shape[1]}", "All dataset fields")
    with c3: metric_card("MISSING VALUES", f"{int(data.isna().sum().sum())}", "Across all columns")

    section("Dataset preview")
    st.dataframe(data.head(20), use_container_width=True, hide_index=True)
    section("Summary statistics")
    st.dataframe(data.describe().T, use_container_width=True)

    left, right = st.columns(2)
    with left:
        col = st.selectbox("Choose a column for distribution", numeric_columns)
        fig = px.histogram(data, x=col, nbins=25, title=f"Distribution of {col}", color_discrete_sequence=["#4f83e8"])
        fig.update_layout(template="plotly_white", height=360, margin=dict(l=20,r=20,t=55,b=20))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        corr = data[numeric_columns].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="Blues", title="Feature correlation")
        fig.update_layout(template="plotly_white", height=360, margin=dict(l=20,r=20,t=55,b=20))
        st.plotly_chart(fig, use_container_width=True)

elif page == "Model Performance":
    st.markdown('<div class="hero"><div class="eyebrow">MODEL EVALUATION</div><h1>Measure model quality.</h1><p>See how closely the model estimates house prices and understand the influence of each feature.</p></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("MAE", f"{mae:,.2f}", "Mean Absolute Error")
    with c2: metric_card("MSE", f"{mse:,.2f}", "Mean Squared Error")
    with c3: metric_card("RMSE", f"{rmse:,.2f}", "Root Mean Squared Error")
    with c4: metric_card("R²", f"{r2:.4f}", "Coefficient of determination")

    actual_pred = pd.DataFrame({"Actual": y_test.to_numpy(), "Predicted": predictions})
    fig = px.scatter(actual_pred, x="Actual", y="Predicted", title="Actual vs Predicted",
                     opacity=.75, color_discrete_sequence=["#3977e8"])
    low = min(actual_pred.min())
    high = max(actual_pred.max())
    fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines",
                             name="Ideal prediction", line=dict(dash="dash", color="#ef8b52")))
    fig.update_layout(template="plotly_white", height=430, margin=dict(l=20,r=20,t=55,b=20))
    st.plotly_chart(fig, use_container_width=True)

    section("Feature coefficients", "A coefficient estimates the change in predicted target for a one-unit increase in that feature, holding other selected features constant.")
    coef = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
    coef["Absolute impact"] = coef["Coefficient"].abs()
    fig = px.bar(coef.sort_values("Absolute impact"), x="Coefficient", y="Feature", orientation="h",
                 title="Linear regression coefficients", color="Coefficient",
                 color_continuous_scale="Blues")
    fig.update_layout(template="plotly_white", height=350, margin=dict(l=20,r=20,t=55,b=20), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Test predictions")
    st.dataframe(actual_pred.head(15).round(2), use_container_width=True, hide_index=True)

elif page == "Price Prediction":
    st.markdown('<div class="hero"><div class="eyebrow">INTERACTIVE PREDICTION</div><h1>Estimate a home value.</h1><p>Adjust the property details below and let your trained Linear Regression model calculate an estimated price.</p></div>', unsafe_allow_html=True)
    st.markdown("#### Enter property details")
    inputs = {}
    cols = st.columns(min(3, len(features)))
    for i, feature in enumerate(features):
        series = pd.to_numeric(clean[feature], errors="coerce").dropna()
        minimum = float(series.min())
        maximum = float(series.max())
        median = float(series.median())
        step = max((maximum - minimum) / 100, 0.1)
        with cols[i % len(cols)]:
            inputs[feature] = st.number_input(
                feature.replace("_", " ").title(),
                min_value=minimum,
                max_value=maximum,
                value=min(max(median, minimum), maximum),
                step=float(step),
                help=f"Observed range in this dataset: {minimum:g} to {maximum:g}"
            )
    if st.button("✨ Predict estimated price", use_container_width=True):
        input_df = pd.DataFrame([inputs], columns=features)
        predicted_price = float(model.predict(input_df)[0])
        st.markdown("### Your estimated property value")
        st.markdown(
            f'<div class="metric-card" style="padding:28px;text-align:center">'
            f'<div class="metric-label">MODEL ESTIMATE</div>'
            f'<div class="metric-value" style="font-size:2.5rem;color:#3977e8">{predicted_price:,.0f}</div>'
            f'<div class="metric-note">Prediction is based on the selected dataset and trained model.</div></div>',
            unsafe_allow_html=True
        )
        st.balloons()
        
    else:
        st.info("Choose the input values and click **Predict estimated price** to see the model output.")

st.markdown('<div class="footer">HOMEVALUE · LINEAR REGRESSION DASHBOARD · AI & ML INTERNSHIP TASK 3</div>', unsafe_allow_html=True)
