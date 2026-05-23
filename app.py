
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import glob
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
.main {
    background-color: #0f1117;
}

h1, h2, h3 {
    color: white;
}

.stButton>button {
    background-color: #7c83ff;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

.metric-box {
    background: #1a1d27;
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource

def load_model():
    model_files = glob.glob('models/*.pkl')

    if not model_files:
        st.error("❌ No trained model found")
        st.stop()

    model = joblib.load(model_files[0])
    return model

model = load_model()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🎓 Student ML Project")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🎯 Prediction",
        "📊 Model Graphs",
        "📈 EDA Analysis",
        "ℹ️ About Project"
    ]
)

# ============================================================
# HOME PAGE
# ============================================================
if page == "🏠 Home":

    st.title("🎓 Student Performance Prediction System")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Project Overview")

        st.write("""
        This Machine Learning project predicts whether a student
        will PASS or FAIL based on:

        - Study Time
        - Previous Grades
        - Absences
        - Family Support
        - Alcohol Consumption
        - Health
        - Social Activities
        - And many more factors
        """)

    with col2:
        st.image("outputs/model_comparison.png", use_container_width=True)

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Models Used", "3")

    with c2:
        st.metric("Dataset", "UCI Student Data")

    with c3:
        st.metric("Prediction Type", "Pass / Fail")

# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "🎯 Prediction":

    st.title("🎯 Student Prediction")

    st.markdown("Enter student details below")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    # ========================================================
    # COLUMN 1
    # ========================================================
    with col1:

        school = st.selectbox("School", ['GP', 'MS'])

        sex = st.selectbox("Gender", ['F', 'M'])

        age = st.slider("Age", 15, 22, 17)

        address = st.selectbox("Address", ['U', 'R'])

        famsize = st.selectbox("Family Size", ['GT3', 'LE3'])

        Pstatus = st.selectbox("Parent Status", ['T', 'A'])

        Medu = st.slider("Mother Education", 0, 4, 2)

        Fedu = st.slider("Father Education", 0, 4, 2)

        Mjob = st.selectbox(
            "Mother Job",
            ['at_home', 'health', 'other', 'services', 'teacher']
        )

        Fjob = st.selectbox(
            "Father Job",
            ['at_home', 'health', 'other', 'services', 'teacher']
        )

    # ========================================================
    # COLUMN 2
    # ========================================================
    with col2:

        reason = st.selectbox(
            "Reason",
            ['course', 'home', 'other', 'reputation']
        )

        guardian = st.selectbox(
            "Guardian",
            ['father', 'mother', 'other']
        )

        traveltime = st.slider("Travel Time", 1, 4, 1)

        studytime = st.slider("Study Time", 1, 4, 2)

        failures = st.slider("Past Failures", 0, 4, 0)

        schoolsup = st.selectbox("School Support", ['yes', 'no'])

        famsup = st.selectbox("Family Support", ['yes', 'no'])

        paid = st.selectbox("Extra Paid Classes", ['yes', 'no'])

        activities = st.selectbox("Activities", ['yes', 'no'])

        nursery = st.selectbox("Nursery", ['yes', 'no'])

        higher = st.selectbox("Higher Education", ['yes', 'no'])

    # ========================================================
    # COLUMN 3
    # ========================================================
    with col3:

        internet = st.selectbox("Internet", ['yes', 'no'])

        romantic = st.selectbox("Romantic", ['yes', 'no'])

        famrel = st.slider("Family Relationship", 1, 5, 3)

        freetime = st.slider("Free Time", 1, 5, 3)

        goout = st.slider("Go Out", 1, 5, 3)

        Dalc = st.slider("Weekday Alcohol", 1, 5, 1)

        Walc = st.slider("Weekend Alcohol", 1, 5, 1)

        health = st.slider("Health", 1, 5, 3)

        absences = st.slider("Absences", 0, 100, 2)

        G1 = st.slider("Grade G1", 0, 20, 10)

        G2 = st.slider("Grade G2", 0, 20, 10)

        course = st.selectbox(
            "Course",
            ['Math', 'Portuguese']
        )

    st.markdown("---")

    # ========================================================
    # ENCODING
    # ========================================================

    cat_maps = {
        'school': {'GP': 0, 'MS': 1},
        'sex': {'F': 0, 'M': 1},
        'address': {'R': 0, 'U': 1},
        'famsize': {'GT3': 0, 'LE3': 1},
        'Pstatus': {'A': 0, 'T': 1},
        'Mjob': {'at_home': 0, 'health': 1, 'other': 2, 'services': 3, 'teacher': 4},
        'Fjob': {'at_home': 0, 'health': 1, 'other': 2, 'services': 3, 'teacher': 4},
        'reason': {'course': 0, 'home': 1, 'other': 2, 'reputation': 3},
        'guardian': {'father': 0, 'mother': 1, 'other': 2},
        'schoolsup': {'no': 0, 'yes': 1},
        'famsup': {'no': 0, 'yes': 1},
        'paid': {'no': 0, 'yes': 1},
        'activities': {'no': 0, 'yes': 1},
        'nursery': {'no': 0, 'yes': 1},
        'higher': {'no': 0, 'yes': 1},
        'internet': {'no': 0, 'yes': 1},
        'romantic': {'no': 0, 'yes': 1},
        'course': {'Math': 0, 'Portuguese': 1},
    }

    # ========================================================
    # PREDICT BUTTON
    # ========================================================

    if st.button("🚀 Predict Student Performance"):

        student = {
            'school': cat_maps['school'][school],
            'sex': cat_maps['sex'][sex],
            'age': age,
            'address': cat_maps['address'][address],
            'famsize': cat_maps['famsize'][famsize],
            'Pstatus': cat_maps['Pstatus'][Pstatus],
            'Medu': Medu,
            'Fedu': Fedu,
            'Mjob': cat_maps['Mjob'][Mjob],
            'Fjob': cat_maps['Fjob'][Fjob],
            'reason': cat_maps['reason'][reason],
            'guardian': cat_maps['guardian'][guardian],
            'traveltime': traveltime,
            'studytime': studytime,
            'failures': failures,
            'schoolsup': cat_maps['schoolsup'][schoolsup],
            'famsup': cat_maps['famsup'][famsup],
            'paid': cat_maps['paid'][paid],
            'activities': cat_maps['activities'][activities],
            'nursery': cat_maps['nursery'][nursery],
            'higher': cat_maps['higher'][higher],
            'internet': cat_maps['internet'][internet],
            'romantic': cat_maps['romantic'][romantic],
            'famrel': famrel,
            'freetime': freetime,
            'goout': goout,
            'Dalc': Dalc,
            'Walc': Walc,
            'health': health,
            'absences': absences,
            'G1': G1,
            'G2': G2,
            'course': cat_maps['course'][course]
        }

        FEATURES = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus',
                    'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
                    'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup',
                    'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic',
                    'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
                    'absences', 'G1', 'G2', 'course']

        X = pd.DataFrame([student])[FEATURES]

        prediction = model.predict(X)[0]

        probability = model.predict_proba(X)[0]

        st.markdown("---")

        st.subheader("📊 Prediction Result")

        c1, c2 = st.columns(2)

        with c1:

            if prediction == 1:
                st.success("✅ STUDENT WILL PASS")
            else:
                st.error("❌ STUDENT MAY FAIL")

        with c2:

            st.info(f"Pass Probability: {probability[1]*100:.2f}%")
            st.warning(f"Fail Probability: {probability[0]*100:.2f}%")

# ============================================================
# MODEL GRAPHS
# ============================================================

elif page == "📊 Model Graphs":

    st.title("📊 ML Model Visualizations")

    st.markdown("---")

    st.subheader("🏆 Model Comparison")
    st.image("outputs/model_comparison.png", use_container_width=True)

    st.markdown("---")

    st.subheader("📈 ROC Curves")
    st.image("outputs/roc_curves.png", use_container_width=True)

    st.markdown("---")

    st.subheader("🎯 Confusion Matrices")
    st.image("outputs/confusion_matrices.png", use_container_width=True)

    st.markdown("---")

    st.subheader("⭐ Feature Importance")
    st.image("outputs/feature_importance.png", use_container_width=True)

# ============================================================
# EDA ANALYSIS
# ============================================================

elif page == "📈 EDA Analysis":

    st.title("📈 Exploratory Data Analysis")

    st.markdown("---")

    st.image("outputs/eda_overview.png", use_container_width=True)

    st.markdown("---")

    st.image("outputs/feature_insights.png", use_container_width=True)

# ============================================================
