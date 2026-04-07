import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Student Performance Prediction")

Age = st.number_input("Age", step=1, format="%d")
Gender = st.number_input("Gender (0/1)", step=1, format="%d")
Ethnicity = st.number_input("Ethnicity", step=1, format="%d")
ParentalEducation = st.number_input("Parental Education", step=1, format="%d")
StudyTimeWeekly = st.number_input("Study Time Weekly", step=0.5)
Absences = st.number_input("Absences", step=1, format="%d")
Tutoring = st.number_input("Tutoring (0/1)", step=1, format="%d")
ParentalSupport = st.number_input("Parental Support", step=1, format="%d")
Extracurricular = st.number_input("Extracurricular (0/1)", step=1, format="%d")
Sports = st.number_input("Sports (0/1)", step=1, format="%d")
Music = st.number_input("Music (0/1)", step=1, format="%d")
Volunteering = st.number_input("Volunteering (0/1)", step=1, format="%d")
GPA = st.number_input("GPA", step=0.1)

if st.button("Predict"):
    input_data = np.array([[Age, Gender, Ethnicity, ParentalEducation,
                            StudyTimeWeekly, Absences, Tutoring, ParentalSupport,
                            Extracurricular, Sports, Music, Volunteering, GPA]])

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)
    st.success(f"Predicted Grade Class: {int(prediction[0])}")
