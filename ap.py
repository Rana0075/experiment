import streamlit as st
import pickle
import numpy as np

# Define constants
MIN_PERCENTAGE = 0
MAX_PERCENTAGE = 100

# Load model
model = pickle.load(open('model.pkl', 'rb'))

def predict_status(secondary_percent, secondary_branch, highschool_percent, high_school_branch, hsc_s, degree_percent, degree_type, mba_percent, work_xp, specialisation, employment_test_percent):
    try:
        # Validate input
        if not (MIN_PERCENTAGE <= secondary_percent <= MAX_PERCENTAGE):
            raise ValueError("Invalid secondary school percentage")
        if not (MIN_PERCENTAGE <= highschool_percent <= MAX_PERCENTAGE):
            raise ValueError("Invalid high school percentage")
        if not (MIN_PERCENTAGE <= degree_percent <= MAX_PERCENTAGE):
            raise ValueError("Invalid degree percentage")
        if not (MIN_PERCENTAGE <= mba_percent <= MAX_PERCENTAGE):
            raise ValueError("Invalid MBA percentage")
        if not (MIN_PERCENTAGE <= employment_test_percent <= MAX_PERCENTAGE):
            raise ValueError("Invalid employment test percentage")

        # Make prediction
        input_data = np.array([[secondary_percent, secondary_branch, highschool_percent, high_school_branch, hsc_s, degree_percent, degree_type, specialisation, mba_percent, work_xp, employment_test_percent]])
        prediction = model.predict_prob(input_data)

        # Output result
        if prediction > 0.5:
            st.success("Higher chances of getting a placement")
        else:
            st.warning("Lower chances of getting a placement")
    except ValueError as e:
        st.error(e)

def main():
    st.title("Campus Placement")
    st.write("Please enter your details:")

    secondary_percent = st.number_input("Secondary school percentage", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE)
    secondary_branch = st.selectbox("Secondary school branch", ["central", "others"])
    highschool_percent = st.number_input("High school percentage", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE)
    high_school_branch = st.selectbox("High school branch", ["central", "others"])
    hsc_s = st.selectbox("High school subject", ["science", "commerce", "arts"])
    degree_percent = st.number_input("Degree percentage", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE)
    degree_type = st.selectbox("Degree type", ["Sci&Tech", "Comm&Mgmt", "others"])
    mba_percent = st.number_input("MBA percentage", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE)
    work_xp = st.selectbox("Work experience", ["yes", "no"])
    specialisation = st.selectbox("Specialisation", ["Mkt&HR", "Mkt&Fin"])
    employment_test_percent = st.number_input("Employment test percentage", min_value=MIN_PERCENTAGE, max_value=MAX_PERCENTAGE)

    if st.button("Predict"):
        predict_status(secondary_percent, secondary_branch, highschool_percent, high_school_branch, hsc_s, degree_percent, degree_type, mba_percent, work_xp, specialisation, employment_test_percent)

if __name__ == "__main__":
    main()
