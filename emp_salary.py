"""
Employee Salary Prediction - Streamlit App
Run:
    pip install -r requirements.txt
    streamlit run app.py

Keep these files in the same folder as app.py:
- employee_salary_dataset.csv
- employee_salary_prediction_model.pkl
- Employee_Salary_Prediction_Results.csv
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "employee_salary_dataset.csv"
MODEL_PATH = BASE_DIR / "employee_salary_prediction_model.pkl"
RESULTS_PATH = BASE_DIR / "Employee_Salary_Prediction_Results.csv"

FEATURES = [
    "Age",
    "Education",
    "Experience_Years",
    "Department",
    "Job_Level",
    "Performance_Rating",
    "Certifications",
    "Overtime_Hours",
    "Remote_Work",
    "City",
    "Company_Tenure",
    "Projects_Completed",
    "Skill_Score",
]
TARGET = "Annual_Salary_LPA"

st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- Styling ---------------------------------------
st.markdown(
    """
    <style>
    /* Bright, high-contrast theme */
    :root {
        --ink: #172554;
        --muted: #263b67;
        --blue: #2563eb;
        --cyan: #06b6d4;
        --purple: #7c3aed;
        --pink: #ec4899;
        --orange: #f97316;
    }

    .stApp {
        background: linear-gradient(135deg, #eef6ff 0%, #f5f0ff 48%, #fff7ed 100%);
        color: var(--ink);
    }

    /* Make formerly gray interface text darker and bold */
    .stApp p, .stApp li, .stApp label,
    .stApp [data-testid="stCaptionContainer"],
    .stApp .stMarkdown, .stApp small,
    .stApp [data-testid="stWidgetLabel"] {
        color: var(--muted) !important;
        font-weight: 700 !important;
    }

    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #172554 !important;
        font-weight: 850 !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #172554 0%, #312e81 55%, #5b21b6 100%);
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    .hero {
        padding: 30px 34px;
        border-radius: 22px;
        color: white;
        background: linear-gradient(115deg, #1d4ed8 0%, #7c3aed 52%, #db2777 100%);
        margin-bottom: 24px;
        box-shadow: 0 12px 28px rgba(79, 70, 229, .22);
        border: 1px solid rgba(255,255,255,.35);
    }
    .hero h1 {
        color: #ffffff !important;
        font-size: 36px;
        font-weight: 900 !important;
        margin: 0 0 9px 0;
        letter-spacing: -.5px;
    }
    .hero p {
        color: #ffffff !important;
        font-size: 16px;
        font-weight: 700 !important;
        opacity: 1;
        margin: 0;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #ffffff 0%, #eaf3ff 100%);
        padding: 18px;
        border-radius: 17px;
        border: 1px solid #bfdbfe;
        border-top: 5px solid #3b82f6;
        box-shadow: 0 5px 16px rgba(37, 99, 235, .12);
    }
    div[data-testid="stMetric"] label {
        color: #1e40af !important;
        font-weight: 800 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #172554 !important;
        font-weight: 900 !important;
    }

    .small-note {
        color: #263b67 !important;
        font-size: 13px;
        font-weight: 800 !important;
    }

    /* Brighter form controls and buttons */
    .stButton > button, .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(100deg, #2563eb, #7c3aed, #db2777) !important;
        color: #ffffff !important;
        border: 0 !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        min-height: 44px;
        box-shadow: 0 5px 13px rgba(79, 70, 229, .2);
    }
    .stButton > button:hover, .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(100deg, #1d4ed8, #6d28d9, #be185d) !important;
        color: #ffffff !important;
        border: 0 !important;
        transform: translateY(-1px);
    }

    [data-baseweb="select"] > div, [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div {
        border-color: #93c5fd !important;
        border-radius: 10px !important;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid #bfdbfe;
        border-radius: 13px;
        overflow: hidden;
    }
    [data-testid="stAlert"] {
        border-radius: 13px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------- Data loading ----------------------------------
@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_results():
    if not RESULTS_PATH.exists():
        return None
    return pd.read_csv(RESULTS_PATH)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, "Model file not found."
    try:
        model = joblib.load(MODEL_PATH)
        expected = list(getattr(model, "feature_names_in_", []))
        if expected and expected != FEATURES:
            return None, "The saved model features do not match this salary dataset."
        return model, None
    except Exception as exc:
        return None, f"Could not load model: {exc}"


data = load_dataset()
results = load_results()
model, model_error = load_model()

if data is None:
    st.error("employee_salary_dataset.csv was not found. Put it in the same folder as app.py.")
    st.stop()

# Use dataset values for dropdown choices, so options stay aligned with the data.
def choices(column, fallback):
    vals = sorted(data[column].dropna().astype(str).unique().tolist())
    return vals or fallback


# --------------------------- Sidebar ---------------------------------------
st.sidebar.markdown("## 💼 Salary Insights")
st.sidebar.caption("Employee Salary Prediction System")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Predict Salary", "Dataset Explorer", "Model Performance"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Model: Random Forest Regression")
st.sidebar.caption("Target: Annual Salary (LPA)")

# --------------------------- Dashboard -------------------------------------
if page == "Dashboard":
    st.markdown(
        """
        <div class="hero">
          <h1>Employee Salary Prediction</h1>
          <p>Explore employee data, understand salary patterns, and estimate annual salary using a trained machine learning model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Employees", f"{len(data):,}")
    c2.metric("Average Salary", f"₹{data[TARGET].mean():.2f} LPA")
    c3.metric("Median Salary", f"₹{data[TARGET].median():.2f} LPA")
    c4.metric("Departments", data["Department"].nunique())

    left, right = st.columns(2)
    with left:
        st.subheader("Salary Distribution")
        fig = px.histogram(
            data, x=TARGET, nbins=30, marginal="box",
            labels={TARGET: "Annual Salary (LPA)"},
            color_discrete_sequence=["#06b6d4"],
        )
        fig.update_layout(template="plotly_white", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Average Salary by Department")
        dept = data.groupby("Department", as_index=False)[TARGET].mean().sort_values(TARGET)
        fig = px.bar(
            dept, x=TARGET, y="Department", orientation="h",
            labels={TARGET: "Average Salary (LPA)"},
            color=TARGET, color_continuous_scale="Turbo",
        )
        fig.update_layout(template="plotly_white", height=360, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Salary by Job Level")
    level = data.groupby("Job_Level", as_index=False)[TARGET].mean()
    fig = px.bar(
        level, x="Job_Level", y=TARGET, color="Job_Level",
        labels={TARGET: "Average Salary (LPA)"},
    )
    fig.update_layout(template="plotly_white", showlegend=False, height=330)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Dataset Records")
    st.dataframe(data.head(10), use_container_width=True, hide_index=True)

# --------------------------- Prediction ------------------------------------
elif page == "Predict Salary":
    st.markdown(
        """
        <div class="hero">
          <h1>Predict an Employee's Salary</h1>
          <p>Enter the employee details below and click Predict Salary.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if model is None:
        st.error(model_error or "The trained model is unavailable.")
        st.info("Check that employee_salary_prediction_model.pkl is in the same folder as app.py.")
        st.stop()

    with st.form("salary_prediction_form"):
        st.subheader("Employee Information")
        a, b, c = st.columns(3)

        with a:
            age = st.number_input("Age", min_value=14, max_value=90, value=28, step=1)
            education = st.selectbox("Education", choices("Education", ["Bachelor", "Master", "Diploma", "PhD"]))
            experience = st.number_input("Experience (Years)", min_value=0.0, max_value=50.0, value=5.0, step=1.0)
            department = st.selectbox("Department", choices("Department", ["IT", "HR", "Sales", "Marketing", "Operations"]))
            job_level = st.selectbox("Job Level", choices("Job_Level", ["Junior", "Mid", "Senior", "Lead", "Manager"]))

        with b:
            performance = st.slider("Performance Rating", 1, 5, 4)
            certifications = st.number_input("Certifications", min_value=0, max_value=30, value=3)
            overtime = st.number_input("Overtime Hours", min_value=0, max_value=100, value=10)
            remote = st.selectbox("Remote Work", choices("Remote_Work", ["Yes", "No"]))

        with c:
            city = st.selectbox("City", choices("City", ["Delhi", "Mumbai", "Chennai", "Hyderabad"]))
            tenure = st.number_input("Company Tenure (Years)", min_value=0, max_value=50, value=3)
            projects = st.number_input("Projects Completed", min_value=0, max_value=100, value=12)
            skill = st.slider("Skill Score", 0.0, 100.0, 80.0, 1.0)

        submitted = st.form_submit_button("✨ Predict Salary", use_container_width=True)

    if submitted:
        employee = pd.DataFrame([{
            "Age": float(age),
            "Education": education,
            "Experience_Years": float(experience),
            "Department": department,
            "Job_Level": job_level,
            "Performance_Rating": int(performance),
            "Certifications": int(certifications),
            "Overtime_Hours": int(overtime),
            "Remote_Work": remote,
            "City": city,
            "Company_Tenure": int(tenure),
            "Projects_Completed": int(projects),
            "Skill_Score": float(skill),
        }], columns=FEATURES)

        try:
            prediction = float(model.predict(employee)[0])
            st.success("Salary prediction completed!")
            st.metric("Estimated Annual Salary", f"₹{prediction:,.2f} LPA")
            st.caption("This is a machine-learning estimate, not a guaranteed salary offer.")
            with st.expander("View input details"):
                st.dataframe(employee, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.caption("Check that your installed scikit-learn version is compatible with the saved model.")

# --------------------------- Dataset Explorer ------------------------------
elif page == "Dataset Explorer":
    st.markdown(
        """
        <div class="hero">
          <h1>Dataset Explorer</h1>
          <p>Filter and inspect the employee salary dataset.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    f1, f2, f3 = st.columns(3)
    with f1:
        selected_dept = st.multiselect("Department", choices("Department", []), default=[])
    with f2:
        selected_level = st.multiselect("Job Level", choices("Job_Level", []), default=[])
    with f3:
        selected_city = st.multiselect("City", choices("City", []), default=[])

    filtered = data.copy()
    if selected_dept:
        filtered = filtered[filtered["Department"].isin(selected_dept)]
    if selected_level:
        filtered = filtered[filtered["Job_Level"].isin(selected_level)]
    if selected_city:
        filtered = filtered[filtered["City"].isin(selected_city)]

    m1, m2, m3 = st.columns(3)
    m1.metric("Filtered Employees", f"{len(filtered):,}")
    m2.metric("Average Salary", f"₹{filtered[TARGET].mean():.2f} LPA" if len(filtered) else "—")
    m3.metric("Average Experience", f"{filtered['Experience_Years'].mean():.1f} years" if len(filtered) else "—")

    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download Filtered CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_employee_salary_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.subheader("Missing Values")
    missing = data.isna().sum().rename("Missing Values").reset_index()
    missing.columns = ["Column", "Missing Values"]
    st.dataframe(missing, use_container_width=True, hide_index=True)

# --------------------------- Model Performance -----------------------------
elif page == "Model Performance":
    st.markdown(
        """
        <div class="hero">
          <h1>Model Performance</h1>
          <p>Review the saved actual-versus-predicted salary results.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if results is None:
        st.warning("Employee_Salary_Prediction_Results.csv was not found.")
        st.stop()

    required = {"Actual Salary", "Predicted Salary"}
    if not required.issubset(results.columns):
        st.error("The results CSV must contain 'Actual Salary' and 'Predicted Salary' columns.")
        st.stop()

    actual = results["Actual Salary"]
    predicted = results["Predicted Salary"]
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    ss_res = float(np.sum((actual - predicted) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Test Records", f"{len(results):,}")
    p2.metric("MAE", f"{mae:.2f} LPA")
    p3.metric("RMSE", f"{rmse:.2f} LPA")
    p4.metric("R² Score", f"{r2:.3f}")

    st.caption("Metrics are calculated from the supplied prediction-results CSV.")
    plot_data = results.copy()
    fig = px.scatter(
        plot_data, x="Actual Salary", y="Predicted Salary",
        labels={"Actual Salary": "Actual Salary (LPA)", "Predicted Salary": "Predicted Salary (LPA)"},
        opacity=0.65, color_discrete_sequence=["#7c3aed"],
        title="Actual Salary vs Predicted Salary",
    )
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))
    fig.add_shape(type="line", x0=low, y0=low, x1=high, y1=high, line=dict(dash="dash", color="#f97316"))
    fig.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prediction Results")
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download Prediction Results",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="employee_salary_prediction_results.csv",
        mime="text/csv",
    )

st.markdown("---")
st.markdown(
    '<p class="small-note">Employee Salary Prediction Project • Built with Python, Streamlit, Pandas and Scikit-learn</p>',
    unsafe_allow_html=True,
)
