import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🎗️",
    layout="wide"
)
st.sidebar.title("About")

st.sidebar.info(
    """
    ## Breast Cancer Detection

    This application uses the K-Nearest Neighbors (KNN) Machine Learning Algorithm to classify breast tumors as:

    ✅ Benign

    ❌ Malignant

    Developed using:
    - Python
    - Streamlit
    - Scikit-learn
    """
)
# ---------------- LOAD MODEL ----------------

model = pickle.load(open("breast_cancer_knn.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
# ---------------- TITLE ----------------

st.title("🎗️ Breast Cancer Detection System")

st.markdown("""
This application predicts whether a breast tumor is:

- 🟢 **Benign (Non-Cancerous)**
- 🔴 **Malignant (Cancerous)**

Please enter the tumor measurements below.
""")

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:

    radius_mean = st.number_input("Radius Mean", value=14.0)

    texture_mean = st.number_input("Texture Mean", value=20.0)

    perimeter_mean = st.number_input("Perimeter Mean", value=90.0)

    area_mean = st.number_input("Area Mean", value=600.0)

    smoothness_mean = st.number_input("Smoothness Mean", value=0.10)

    compactness_mean = st.number_input("Compactness Mean", value=0.10)

    concavity_mean = st.number_input("Concavity Mean", value=0.08)

    concave_points_mean = st.number_input("Concave Points Mean", value=0.04)

    symmetry_mean = st.number_input("Symmetry Mean", value=0.18)

    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", value=0.06)
with col2:

    radius_se = st.number_input("Radius SE", value=0.40)

    texture_se = st.number_input("Texture SE", value=1.20)

    perimeter_se = st.number_input("Perimeter SE", value=2.80)

    area_se = st.number_input("Area SE", value=40.0)

    smoothness_se = st.number_input("Smoothness SE", value=0.007)

    compactness_se = st.number_input("Compactness SE", value=0.025)

    concavity_se = st.number_input("Concavity SE", value=0.030)

    concave_points_se = st.number_input("Concave Points SE", value=0.012)

    symmetry_se = st.number_input("Symmetry SE", value=0.020)

    fractal_dimension_se = st.number_input("Fractal Dimension SE", value=0.003)
with col3:

    radius_worst = st.number_input("Radius Worst", value=16.0)

    texture_worst = st.number_input("Texture Worst", value=25.0)

    perimeter_worst = st.number_input("Perimeter Worst", value=105.0)

    area_worst = st.number_input("Area Worst", value=800.0)

    smoothness_worst = st.number_input("Smoothness Worst", value=0.14)

    compactness_worst = st.number_input("Compactness Worst", value=0.25)

    concavity_worst = st.number_input("Concavity Worst", value=0.30)

    concave_points_worst = st.number_input("Concave Points Worst", value=0.12)

    symmetry_worst = st.number_input("Symmetry Worst", value=0.28)

    fractal_dimension_worst = st.number_input("Fractal Dimension Worst", value=0.08)
# ---------------- PREDICT BUTTON ----------------

if st.button("Predict"):

    # Create input array
    input_data = np.array([[
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean,
        compactness_mean,
        concavity_mean,
        concave_points_mean,
        symmetry_mean,
        fractal_dimension_mean,

        radius_se,
        texture_se,
        perimeter_se,
        area_se,
        smoothness_se,
        compactness_se,
        concavity_se,
        concave_points_se,
        symmetry_se,
        fractal_dimension_se,

        radius_worst,
        texture_worst,
        perimeter_worst,
        area_worst,
        smoothness_worst,
        compactness_worst,
        concavity_worst,
        concave_points_worst,
        symmetry_worst,
        fractal_dimension_worst
    ]])

    # Scale the input data
    input_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data)

    # Get prediction probability
    probability = model.predict_proba(input_data)

    # Calculate confidence
    confidence = np.max(probability) * 100

    # Display result
    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("🔴 **Malignant (Cancerous)**")
    else:
        st.success("🟢 **Benign (Non-Cancerous)**")

    # Display confidence
    st.info(f"Confidence: {confidence:.2f}%")

    # Progress bar
    st.progress(int(confidence))

# Footer
st.markdown("---")
st.caption("Developed using Python, Streamlit & Scikit-learn")