import streamlit as st
import numpy as np
import pickle
import time  # For a progress animation

# Load the trained model
try:
    model = pickle.load(open("diabetes_model.sav", "rb"))
except FileNotFoundError:
    st.error("Model file 'diabetes_model.sav' not found. Please ensure it is in the same directory as this script.")
    st.stop()

# Streamlit UI
st.title("Diabetes Prediction App")
st.image("Diabetes.png", caption="Diabetes Prediction", use_column_width=True)
st.markdown("Enter your details below to assess your diabetes risk. All fields are required.")

# Input form for better organization
with st.form(key="diabetes_form"):
    # Input fields with appropriate widgets and constraints
    Pregnancies = st.number_input(
        "Number of Pregnancies", min_value=0, max_value=20, value=0, step=1, help="Enter the number of times pregnant."
    )
    Glucose = st.number_input(
        "Glucose Level (mg/dL)", min_value=0.0, max_value=200.0, value=0.0, step=0.1, help="Plasma glucose concentration."
    )
    BloodPressure = st.number_input(
        "Blood Pressure (mmHg)", min_value=0.0, max_value=150.0, value=0.0, step=0.1, help="Diastolic blood pressure."
    )
    SkinThickness = st.number_input(
        "Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.1, help="Triceps skin fold thickness."
    )
    Insulin = st.number_input(
        "Insulin Level (mu U/ml)", min_value=0.0, max_value=1000.0, value=0.0, step=0.1, help="2-Hour serum insulin."
    )
    BMI = st.number_input(
        "BMI (kg/m²)", min_value=0.0, max_value=70.0, value=0.0, step=0.1, help="Body Mass Index."
    )
    DiabetesPedigreeFunction = st.number_input(
        "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.0, step=0.01, 
        help="A function that scores likelihood of diabetes based on family history."
    )
    Age = st.number_input(
        "Age (years)", min_value=0, max_value=120, value=0, step=1, help="Your age in years."
    )

    # Submit button for the form
    submit_button = st.form_submit_button(label="Predict")

# Prediction logic
if submit_button:
    # Validate inputs
    if any(v == 0 for v in [Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]):
        st.warning("Please fill in all fields with valid values (greater than 0 where applicable).")
    else:
        # Prepare input data as a numpy array
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]], dtype=float)

        # Add a progress bar for interactivity
        with st.spinner("Analyzing your data..."):
            time.sleep(1)  # Simulate processing time for effect
            try:
                # Make prediction
                prediction = model.predict(input_data)[0]

                # Display result with enhanced feedback
                if prediction == 1:
                    st.error("🚨 High Risk of Diabetes Detected!")
                    st.markdown("Consider consulting a healthcare professional for further evaluation.")
                else:
                    st.success("✅ Low Risk of Diabetes!")
                    st.markdown("Keep maintaining a healthy lifestyle!")
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")

# Additional interactivity: Sidebar with info
st.sidebar.header("About This App")
st.sidebar.markdown("""
This app uses a pre-trained machine learning model to predict diabetes risk based on your input. The model was trained on a dataset with features like glucose levels, BMI, and age. Ensure all inputs are accurate for the best results.
""")