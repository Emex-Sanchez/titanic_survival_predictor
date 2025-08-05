import streamlit as st
import numpy as np
import joblib

model = joblib.load('titanic_model.pkl')  # Load your trained model

st.title("🚢 Titanic Survival Predictor")

st.markdown("""
Welcome to the Titanic Survival Predictor!  
This app uses a **Machine Learning model (Logistic Regression)** trained on the Titanic dataset to predict whether a passenger would have survived the Titanic disaster.

 💡 How it works:
- You provide passenger details (class, sex, age, fare, etc.)
- The model calculates the probability of survival
- Based on the prediction, the app tells you if the person **survived (1)** or **did not survive (0)**

> This project was built using `Python`, `pandas`, `scikit-learn`, and `Streamlit`.
""")

# Input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, step=1)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, step=1)
fare = st.number_input("Fare Paid", min_value=0.0, step=0.1)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Convert inputs to numeric
sex = 0 if sex == "male" else 1
embarked = {"C": 0, "Q": 1, "S": 2}[embarked]

# Prediction
features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
if st.button("Predict"):
    prediction = model.predict(features)[0]
    st.success("✅ Survived" if prediction == 1 else "❌ Did not survive")
