import streamlit as st
import joblib as jb
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load model
model_path = "model_decisiontree.joblib"
model = jb.load(model_path)

st.title('Decision Tree Model Prediction')

# Input fields
gender = st.selectbox('Jenis Kelamin : ', options=['Male', 'Female'])
age = st.number_input('Umur Pendaftar : ', min_value=18, max_value=100, value=30)
marital_status = st.selectbox('Status Perkawinan : ', options=['Single', 'Married', 'Widower', 'Divorced', 'Facto Union', 'Legally Separated'])
scholarship_holder = st.selectbox('Penerima Beasiswa : ', options=['Yes', 'No'])
course = st.selectbox('Jurusan : ', options=['Biofuel Production Technologies', 'Animation and Multimedia Design', 'Social Service (evening attendance)',
                                             'Agronomy', 'Communication Design', 'Veterinary Nursing', 'Informatics Engineering', 'Equinculture',
                                             'Management', 'Social Service', 'Nursing', 'Tourism', 'Oral Hygiene',
                                             'Advertising and Marketing Management', 'Journalism and Communication',
                                             'Basic Education', 'Management (evening attendance)'])
international = st.selectbox('International : ', options=['Yes', 'No'])
daytime_evening_attendance = st.selectbox('Daytime_evening_attendance : ', options=['Daytime', 'Evening'])
educational_special_needs = st.selectbox('Educational_special_needs : ', options=['Yes', 'No'])

# Dictionary mapping
mapping_dict = {
    'gender': {'Male': 1, 'Female': 0},
    'marital_status': {'Single': 1, 'Married': 2, 'Widower': 3, 'Divorced': 4, 'Facto Union': 5, 'Legally Separated': 6},
    'scholarship_holder': {'Yes': 1, 'No': 0},
    'course': {
        'Biofuel Production Technologies': 0, 'Animation and Multimedia Design': 1, 'Social Service (evening attendance)': 2,
        'Agronomy': 3, 'Communication Design': 4, 'Veterinary Nursing': 5, 'Informatics Engineering': 6, 'Equinculture': 7,
        'Management': 8, 'Social Service': 9, 'Nursing': 10, 'Tourism': 11, 'Oral Hygiene': 12, 'Advertising and Marketing Management': 13,
        'Journalism and Communication': 14, 'Basic Education': 15, 'Management (evening attendance)': 16
    },
    'international': {'Yes': 1, 'No': 0},
    'daytime_evening_attendance': {'Daytime': 1, 'Evening': 0},
    'educational_special_needs': {'Yes': 1, 'No': 0}
}

# Convert categorical to numerical
gender = mapping_dict['gender'][gender]
marital_status = mapping_dict['marital_status'][marital_status]
scholarship_holder = mapping_dict['scholarship_holder'][scholarship_holder]
course = mapping_dict['course'][course]
international = mapping_dict['international'][international]
daytime_evening_attendance = mapping_dict['daytime_evening_attendance'][daytime_evening_attendance]
educational_special_needs = mapping_dict['educational_special_needs'][educational_special_needs]

input_data = [gender, age, marital_status, scholarship_holder, course, international, daytime_evening_attendance, educational_special_needs]
input_array = np.array([input_data])

if st.button('Predict'):
    prediction = model.predict(input_array)
    if prediction[0] == 0:
        st.write('Prediction: Not Dropout')
    else:
        st.write('Prediction: Dropout')