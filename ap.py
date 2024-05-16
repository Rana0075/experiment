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
    html="""
    <div style="background-color:#025246;padding:10px">
    <h2 style="color:white;text-align:center;">Placement prediction ml</h2>
    </div>
    """
    st.markdown(html,prediction_html=True)
    secondary_percent =st.text_input("secondary_percent","Type Here")
    secondary_branch =st.<select name ='ssb'> <option value="central"> central</option> <option value="others">others</option>
    highschool_percent = st.text_input("secondary_percent","Type Here")
    degree_percent = st.text_input("secondary_percent","Type Here")
    degree_type =st.<select name ='degree_type'> <option value="Sci&Tech"> Sci&Tech</option> <option value="Comm&Mgmt">Comm&Mgmt</option> <option value="others">others</option>
    mba_percent = st.text_input("secondary_percent","Type Here")
    work_xp =st.<select name ='work_xp'> <option value="Yes"> yes</option> <option value="No">no</option>
    employment_test_percent = st.text_input("secondary_percent","Type Here")
    high_school_branch=st.<select name ='hsb'> <option value="central"> central</option> <option value="others">others</option>
    hsc_s=st.<select name ='hsc_s'> <option value="science"> science</option> <option value="commerce">commerce</option> <option value="Arts">arts </option>
    specialisation=st.<select name ='specialisation'> <option value="Mkt&HR"> Mkt&HR</option> <option value="Mkt&Fin">Mkt&Fin</option>
    
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
            

if _name_ == 'main':
    main()
