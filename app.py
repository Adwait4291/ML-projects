import streamlit as st
import pandas as pd
import sys
import os

# Add the 'src' directory to the Python path
# This ensures that the application can find the 'src' module
# Adjust the path if your streamlit_app.py is located differently relative to 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    # Assuming 'src' is in the same directory or added to path correctly
    from pipelines.predict_pipeline import CustomData, PredictPipeline
    from exceptions import CustomException
except ImportError as e:
    st.error(f"Error importing modules: {e}. Make sure 'src' directory is in the Python path and contains necessary files (pipelines/predict_pipeline.py, exceptions.py).")
    st.stop() # Stop execution if imports fail

# Streamlit App Title
st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title("Student Math Score Prediction")
st.write("Enter the student's details below to predict their Math Score.")

# --- Input Form ---
st.header("Student Details")

# Create columns for layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender",
        options=['male', 'female'],
        index=0 # Default selection
    )
    race_ethnicity = st.selectbox(
        "Race/Ethnicity",
        options=['group A', 'group B', 'group C', 'group D', 'group E'],
        index=2 # Default selection
    )
    parental_level_of_education = st.selectbox(
        "Parental Level of Education",
        options=["bachelor's degree", "some college", "master's degree", "associate's degree", "high school", "some high school"],
        index=1 # Default selection
    )

with col2:
    lunch = st.selectbox(
        "Lunch Type",
        options=["standard", "free/reduced"],
        index=0 # Default selection
    )
    test_preparation_course = st.selectbox(
        "Test Preparation Course",
        options=["none", "completed"],
        index=0 # Default selection
    )

# Use columns for scores as well for better alignment
col3, col4 = st.columns(2)

with col3:
    reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=50, step=1)

with col4:
    writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=50, step=1)

# --- Prediction ---
if st.button("Predict Math Score", type="primary"):
    try:
        # Create CustomData instance
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )

        # Get data as DataFrame
        pred_df = data.get_data_as_data_frame()
        st.write("Input Data:")
        st.dataframe(pred_df)

        # Instantiate prediction pipeline and predict
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Display results
        st.success(f"Predicted Math Score: {results[0]:.2f}")

    except CustomException as e:
        st.error(f"An error occurred during prediction: {e}")
    except FileNotFoundError:
        st.error("Error: Model or preprocessor file not found. Make sure 'artifacts/model.pkl' and 'artifacts/preprocessor.pkl' exist.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# --- Sidebar Information Removed ---
# st.sidebar.header("About the Model")
# st.sidebar.info(
#     """
#     ... (previous content removed) ...
#     """
# )