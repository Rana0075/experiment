import streamlit as st
import pickle
import numpy as np
model= pickle.load(open('model.pkl','rb'))

def predict_status(secondary_percent ,secondary_branch, highschool_percent,high_school_branch,hsc_s,degree_percent,degree_type,mba_percent,work_xp,specialisation,employment_test_percent):
    input=np.array([[secondary_percent ,secondary_branch, highschool_percent,high_school_branch,hsc_s,degree_percent,degree_type ,specialisation,
          mba_percent,work_xp,employment_test_percent]]).astype(np.float64)
    prediction=model.predict_prob(input)
    return prediction

def main():
    st.title("campus placement")
    html= """
    <div style="background-color:#025246;padding:10px">
    <h2 style="color:white;text-align:center;">Placement prediction ml</h2>
    </div>
    """
    st.markdown(html,prediction_html=True)
    secondary_percent = st.text_input("mention the percentage u received in secondary school:") 
    secondary_branch = st.selectbox("pick the branch you chose in secondary school:" ,  "central","others")
    highschool_percent =st.text_input("percentage scored in highschool:")
    high_school_branch = st.selectbox("mention the branch you chose in highschool :" ,"central","others")
    hsc_s = st.selectbox("pick the subject you chose in high school:" , "science","commerce","Arts")
    degree_percent = st.text_input("mention the percentage you recieved in college :")
    degree_type = st.selectbox("pick the subject for degree:" ,'degree_type',"Sci&Tech","Comm&Mgmt","others")
    work_xp = st.selectbox("Do you have any work experience:" , "yes","no")
    employment_test_percent = st.text_input("mention your percentage in employment test")
    specialisation = st.selectbox("mention your specialization:" ,"Mkt&HR","Mkt&Fin" )
    mba_percent = st.text_input("mention your mba percentage :")

placed_html="""
<div style="background-color:#F4D03F;padding:10px">
<h2 style="color:white;text-align:center;"> Higher chances of getting a placement</h2>
</div>
"""
not_placed_html="""
<div style="background-color:#F08080;padding:10px">
<h2 style="color:black;text-align:center;"> Low chances of getting a placement</h2>
</div>
"""

if st.button("Predict"):
    output=predict_status(secondary_percent ,secondary_branch, highschool_percent,high_school_branch,hsc_s,degree_percent,degree_type,mba_percent,work_xp,specialisation,employment_test_percent)
    st.success('Result of prediction {}'.format(output))
        
    if output==0:
        st.markdown(not_placed_html,prediction_html=True)
    else:
        st.markdown(placed_html,prediction_html=True)
            

if __name__== '__main__':
    main()
