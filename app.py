import streamlit as st
import numpy as np
import joblib
# Step 1: Load the saved model
model = joblib.load('C://Users//muzri//Documents//module4 mechine learning//feature engeering//titanic surviver data cleaning//notebooks//titanic_model.pkl')
# Step 2: Create input fields in Streamlit
st.title("Titanic Survival Prediction")
PassengerId = st.number_input("Passenger ID (optional)",min_value=0,max_value=4000,value=0)
Pclass = st.selectbox("Pclass (1=First, 2=Second, 3=Third)", [1, 2, 3])
Sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
Age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
SibSp = st.number_input("SibSp", min_value=0, max_value=10, value=0)
Parch = st.number_input("Parch", min_value=0, max_value=10, value=0)
Fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=72.0)
Familysize = st.number_input("Familysize", min_value=0, max_value=10, value=1)
IsAlone = st.selectbox("IsAlone (0=No, 1=Yes)", [0, 1])
Title = st.number_input("Title", min_value=0, max_value=5, value=2)
Embarked_C = st.selectbox("Embarked_C", [0, 1])
Embarked_Q = st.selectbox("Embarked_Q", [0, 1])
Embarked_S = st.selectbox("Embarked_S", [0, 1])
# Step 3: Predict button
if st.button("Predict"):
    # Create input array in correct order
    input_data = np.array([[PassengerId,Pclass, Sex, Age, SibSp, Parch, Fare,
                            Familysize, IsAlone, Title, Embarked_C, Embarked_Q, Embarked_S]])
    
    # Predict class
    prediction = model.predict(input_data)[0]
    
    # Predict probability
    probability = model.predict_proba(input_data)[0, 1]
    
    st.write(f"Predicted Class: {'Survived' if prediction==1 else 'Not Survived'}")
    st.write(f"Survival Probability: {probability:.2f}")




