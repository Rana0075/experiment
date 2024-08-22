# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import pickle
import warnings

# Load data from CSV file
try:
    data = pd.read_csv("D:/python/git/experiment/train.csv")
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit()

# Drop unnecessary columns
# Dropping 'salary', 'sl_no', and 'gender' as they are not relevant to the analysis
data = data.drop(['salary', 'sl_no', 'gender'], axis=1)

# Create dummy variables for categorical features
# Using pd.get_dummies to create dummy variables for categorical features
data2 = pd.get_dummies(data, drop_first=True)

# Split data into features and target variable
y = data2.status_Placed
x = data2.drop(['status_Placed'], axis=1)

# Split data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=3)

# Train Linear Discriminant Analysis model
# Using Linear Discriminant Analysis as it is suitable for classification problems
lda = LinearDiscriminantAnalysis()
lda.fit(xtrain, ytrain)

# Save trained model
# Saving the trained model for future use
pickle.dump(lda, open('model.pkl', 'wb'))


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

        # Convert categorical variables to numerical variables
        secondary_branch_map = {"central": 0, "others": 1}
        high_school_branch_map = {"central": 0, "others": 1}
        hsc_s_map = {"science": 0, "commerce": 1, "arts": 2}
        degree_type_map = {"Sci&Tech": 0, "Comm&Mgmt": 1, "others": 2}
        work_xp_map = {"yes": 0, "no": 1}
        specialisation_map = {"Mkt&HR": 0, "Mkt&Fin": 1}

        secondary_branch = secondary_branch_map[secondary_branch]
        high_school_branch = high_school_branch_map[high_school_branch]
        hsc_s = hsc_s_map[hsc_s]
        degree_type = degree_type_map[degree_type]
        work_xp = work_xp_map[work_xp]
        specialisation = specialisation_map[specialisation]

        # Make prediction
        input_data = np.array([[secondary_percent, secondary_branch, highschool_percent, high_school_branch, hsc_s, degree_percent, degree_type, specialisation, mba_percent, work_xp, employment_test_percent]])
        prediction = model.predict_proba(input_data)[0][1]

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
