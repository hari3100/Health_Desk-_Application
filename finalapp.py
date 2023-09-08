import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import warnings
from PIL import Image

warnings.filterwarnings("ignore")

# Load model files and scalers
calories_model = pickle.load(open('calmodel.pkl', 'rb'))
h1n1_model = pickle.load(open('h1n1_model.pkl', 'rb'))
h1n1_scaler = pickle.load(open('h1n1_scaler.pkl', 'rb'))
diabetes_model = pickle.load(open('diabetes_model.pkl', 'rb'))
diabetes_scaler = pickle.load(open('diabetes_scaler.pkl', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.pkl', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_disease_model.pkl', 'rb'))
parkinsons_scaler = pickle.load(open('scaler_parkinsons.pkl', 'rb'))

# CSS styling
css = """
<style>
body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f5f5f5;
}
header {
    background-color: #0066cc;
    color: white;
    padding: 20px;
    text-align: center;
    font-size: 2.2rem;
    font-weight: bold;
}
.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    background-color: white;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.subheader {
    font-size: 1.5rem;
    margin-bottom: 20px;
}
.feature-label {
    font-weight: bold;
    margin-bottom: 5px;
}
.custom-button {
    background-color: #0066cc;
    color: white;
    font-size: 1.2rem;
    padding: 10px 25px;
    border-radius: 5px;
    margin-top: 30px;
    cursor: pointer;
    transition: background-color 0.3s ease, transform 0.3s ease;
}
.custom-button:hover {
    background-color: #0059b3;
    transform: scale(1.05);
}
.result-image {
    margin-top: 20px;
    max-width: 100%;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.footer {
    color: #666666;
    text-align: center;
    margin-top: 50px;
}
</style>
"""

# Function to handle Diabetes Prediction
def diabetes_prediction():
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.subheader('Diabetes Prediction using Machine Learning')

    st.write('Please enter the following details:')
    
    # Input fields
    pregnancies = st.text_input('Number of Pregnancies')
    glucose = st.text_input('Glucose Level')
    blood_pressure = st.text_input('Blood Pressure value')
    skin_thickness = st.text_input('Skin Thickness value')
    insulin = st.text_input('Insulin Level')
    bmi = st.text_input('BMI value')
    diabetes_pedigree_function = st.text_input('Diabetes Pedigree Function value')
    age = st.text_input('Age of the Person')
    
    # Check if all input fields are not empty
    if pregnancies and glucose and blood_pressure and skin_thickness and insulin and bmi and diabetes_pedigree_function and age:
        # preprocess input using StandardScaler
        input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]
        input_data_scaled = diabetes_scaler.transform(input_data)

        # code for Prediction
        if st.button('Diabetes Test Result'):
            diab_prediction = diabetes_model.predict(input_data_scaled)
            if diab_prediction[0] == 1:
                st.success('The person is diabetic')
                st.image('diabetes_positive_image.jpg', caption='Diabetes Positive')
            else:
                st.success('The person is not diabetic')
                st.image('diabetes_negative_image.jpg', caption='Diabetes Negative')
    else:
        st.warning('Please fill in all the input fields.')
    
    st.markdown('</div>', unsafe_allow_html=True)

# Function to handle Heart Disease Prediction
def heart_disease_prediction():
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.subheader('Heart Disease Prediction using Machine Learning')

    st.write('Please enter the following details:')
    
    # Input fields
    age = st.number_input('Age', min_value=1, max_value=100, value=50)
    sex_mapping = {'Male': 1, 'Female': 0}
    sex = st.selectbox('Sex', list(sex_mapping.keys()))
    sex_val = sex_mapping[sex]
    
    cp_mapping = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 3}
    cp = st.selectbox('Chest Pain Type', list(cp_mapping.keys()))
    cp_val = cp_mapping[cp]
    
    trestbps = st.number_input('Resting Blood Pressure', min_value=50, max_value=250, value=120)
    chol = st.number_input('Serum Cholestoral in mg/dl', min_value=50, max_value=600, value=200)
    
    fbs_mapping = {'No': 0, 'Yes': 1}
    fbs_val = st.selectbox('Fasting Blood Sugar > 120 mg/dl', list(fbs_mapping.keys()))
    fbs_val = fbs_mapping[fbs_val]
    
    restecg_mapping = {'Normal': 0, 'ST-T Wave Abnormality': 1, 'Probable or Definite LVH': 2}
    restecg = st.selectbox('Resting Electrocardiographic Results', list(restecg_mapping.keys()))
    restecg_val = restecg_mapping[restecg]
    
    thalach = st.number_input('Maximum Heart Rate achieved', min_value=50, max_value=220, value=150)
    
    exang = st.radio('Exercise Induced Angina', ['No', 'Yes'])
    exang_val = 1 if exang == 'Yes' else 0
    
    oldpeak = st.number_input('ST depression induced by exercise', value=1.0)
    
    slope_mapping = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', list(slope_mapping.keys()))
    slope_val = slope_mapping[slope]
    
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', [0, 1, 2, 3])
    
    thal_mapping = {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2}
    thal = st.selectbox('Thalassemia', list(thal_mapping.keys()))
    thal_val = thal_mapping[thal]

    if st.button('Predict Heart Disease'):
        input_data = [[age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val, thalach, exang_val, oldpeak, slope_val, ca, thal_val]]
        heart_prediction = heart_disease_model.predict(input_data)
        
        if heart_prediction[0] == 0:
            st.success('The person is predicted to not have heart disease')
            st.image('heart_negative_image.jpg', use_column_width=True)
        else:
            st.success('The person is predicted to have heart disease')
            st.image('heart_positive_image.jpg', use_column_width=True)
    
    
    st.markdown('</div>', unsafe_allow_html=True)

# Function to handle Parkinson's Prediction
def parkinsons_prediction():
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.subheader("Parkinson's Disease Prediction using Machine Learning")

    st.write('Please enter the following details:')
    
    # Input fields
    fo = st.text_input('MDVP Fo(Hz)')
    fhi = st.text_input('MDVP Fhi(Hz)')
    flo = st.text_input('MDVP Flo(Hz)')
    jitter_percent = st.text_input('MDVP Jitter(%)')
    jitter_abs = st.text_input('MDVP Jitter(Abs)')
    rap = st.text_input('MDVP (RAP)')
    ppq = st.text_input('MDVP (PPQ)')
    ddp = st.text_input('Jitter (DDP)')
    shimmer = st.text_input('MDVP (Shimmer)')
    shimmer_db = st.text_input('MDVP Shimmer(dB)')
    apq3 = st.text_input('Shimmer (APQ3)')
    apq5 = st.text_input('Shimmer (APQ5)')
    apq = st.text_input('MDVP (APQ)')
    dda = st.text_input('Shimmer (DDA)')
    nhr = st.text_input('NHR')
    hnr = st.text_input('HNR')
    rpde = st.text_input('RPDE')
    dfa = st.text_input('DFA')
    spread1 = st.text_input('spread1')
    spread2 = st.text_input('spread2')
    d2 = st.text_input('D2')
    ppe = st.text_input('PPE')
    
    # Check if all input fields are not empty
    if fo and fhi and flo and jitter_percent and jitter_abs and rap and ppq and ddp and shimmer and shimmer_db and apq3 and apq5 and apq and dda and nhr and hnr and rpde and dfa and spread1 and spread2 and d2 and ppe:
        # preprocess input using StandardScaler
        input_data = [[fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]]
        input_data_scaled = parkinsons_scaler.transform(input_data)

        # code for Prediction
        if st.button("Parkinson's Test Result"):
            parkinsons_prediction = parkinsons_model.predict(input_data_scaled)
            if parkinsons_prediction[0] == 1:
                st.success("The person has Parkinson's disease")
                st.image('parkinsons_positive_image.jpg', caption="Seek help from professionals")
            else:
                st.success("The person does not have Parkinson's disease")
                st.image('parkinsons_negative_image.jpg', caption="You're safe, Take care !")
    else:
        st.warning('Please fill in all the input fields.')
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main function for the app
def main():
    st.set_page_config(
        page_title="Health Desk App",
        #layout="wide",
        initial_sidebar_state="collapsed"
    )
    st.markdown(css, unsafe_allow_html=True)
    st.markdown('<header>Health Desk App</header>', unsafe_allow_html=True)

    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.write('Welcome to the Health Desk App! Choose a prediction type below:')

    # Prediction type buttons with images and highlighting effect
    prediction_type = st.selectbox(
        'Select a Prediction',
        [
            ('Calories Burned', 'calories_image.jpg'),
            ('H1N1 Vaccine Usage', 'h1n1_image.jpg'),
            ('Diabetes Prediction', 'diabetes_image.jpg'),
            ('Heart Disease Prediction', 'heart_image.jpg'),
            ("Parkinsons Prediction", 'parkinsons_image.jpg')
        ],
        format_func=lambda x: x[0]
    )

    prediction_name, prediction_image = prediction_type
    prediction_name = prediction_name.replace(" ", "_").lower()
    prediction_image_path = f'images/{prediction_image}'

    st.image(prediction_image_path, use_column_width=True)
    st.markdown(f'<div class="custom-button">{prediction_name.replace("_", " ").title()}</div>', unsafe_allow_html=True)

    if prediction_name == 'calories_burned':
        # Title
        st.title("Calories Burned Prediction App")

        # Input form
        st.header("Enter the details:")
        gender = st.selectbox("Gender", ['Male', 'Female'])
        gender = 1 if gender == 'Male' else 0
        age = st.number_input("Age", min_value=1, max_value=150, step=1)
        height = st.number_input("Height (cm)", min_value=1, max_value=300, step=1)
        weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, step=0.1)
        duration = st.number_input("Duration (minutes)", min_value=1, max_value=1440, step=1)
        heart_rate = st.number_input("Heart Rate", min_value=1, max_value=250, step=1)
        body_temp = st.number_input("Body Temperature (Celsius)", min_value=30.0, max_value=50.0, step=0.1)

        # Make prediction
        input_data = pd.DataFrame({
            'Gender': [gender],
            'Age': [age],
            'Height': [height],
            'Weight': [weight],
            'Duration': [duration],
            'Heart_Rate': [heart_rate],
            'Body_Temp': [body_temp]
        })

        if st.button("Predict Calories Burned"):
            prediction = calories_model.predict(input_data)
            st.success(f"Predicted Calories Burned: {prediction[0]:.2f} kcal")
        
        pass
    elif prediction_name == 'h1n1_vaccine_usage':
        st.markdown(
        """
        <style>
        body {
            font-family: 'Arial', sans-serif;
            color: #333333;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #0066cc;
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 2.2rem;
            font-weight: bold;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .feature-label {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .custom-button {
            background-color: #0066cc;
            color: white;
            font-size: 1.2rem;
            padding: 10px 25px;
            border-radius: 5px;
            margin-top: 30px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .custom-button:hover {
            background-color: #0059b3;
        }
        .footer {
            color: #666666;
            text-align: center;
            margin-top: 50px;
        }
        </style>
        """,
        unsafe_allow_html=True
        )

        st.markdown('<div class="header">Vaccine Usage Prediction App</div>', unsafe_allow_html=True)
        st.subheader('Predict whether someone received the H1N1 vaccine')

        st.markdown('<div class="container">', unsafe_allow_html=True)

        st.write('Enter the values for the features below:')

        
        h1n1_worry = st.slider('H1N1 Worry', 0, 3, 1, key='h1n1_worry')
        h1n1_awareness = st.slider('H1N1 Awareness', 0, 2, 1, key='h1n1_awareness')
        dr_recc_h1n1_vacc = st.checkbox('Doctor Recommended H1N1 Vaccine', key='dr_recc_h1n1_vacc')
        dr_recc_seasonal_vacc = st.checkbox('Doctor Recommended Seasonal Flu Vaccine', key='dr_recc_seasonal_vacc')
        chronic_medic_condition = st.checkbox('Has Chronic Medical Condition', key='chronic_medic_condition')
        is_health_worker = st.checkbox('Is Health Worker', key='is_health_worker')
        is_h1n1_vacc_effective = st.select_slider('Perceived H1N1 Vaccine Effectiveness', options=[1, 2, 3, 4, 5], value=3, key='is_h1n1_vacc_effective')
        is_h1n1_risky= st.select_slider('Perceived H1N1 Risk', options=[1, 2, 3, 4, 5], value=3,key='is_h1n1_risky')
        is_seas_vacc_effective=st.select_slider('Perceived Seasonal Vaccine Effectiveness', options=[1, 2, 3, 4, 5], value=3,key='is_seas_vacc_effective')
        is_seas_risky=st.select_slider('Perceived Seasonal Flu Risk', options=[1, 2, 3, 4, 5], value=3,key='is_seas_risky')



        if st.button('Predict', key='predict_btn'):
            # Create a pandas DataFrame from the input data
            data = pd.DataFrame({
        'h1n1_worry': [h1n1_worry],
        'h1n1_awareness': [h1n1_awareness],
        'dr_recc_h1n1_vacc': [dr_recc_h1n1_vacc],
        'dr_recc_seasonal_vacc': [dr_recc_seasonal_vacc],
        'chronic_medic_condition': [chronic_medic_condition],
        'is_health_worker': [is_health_worker],
        'is_h1n1_vacc_effective': [is_h1n1_vacc_effective],
        'is_h1n1_risky': [is_h1n1_risky],
        'is_seas_vacc_effective': [is_seas_vacc_effective],
        'is_seas_risky': [is_seas_risky]
        })

            # Scale the input data using the loaded scaler
            scaled_data = h1n1_scaler.transform(data)

            # Make predictions using the classifier
            result = h1n1_model.predict(scaled_data)

            if result[0] == 1:
                st.write("The respondent received the H1N1 vaccine.")
                st.image('H1N1vaccine.jpg', use_column_width=True, caption='Vaccinated!')
            else:
                st.write("The respondent did not receive the H1N1 vaccine.")
                st.image('Flu-vaccine.jpg', use_column_width=True, caption='Take a precaustionary flu vaccine!')

        
        st.markdown('</div>', unsafe_allow_html=True)

        pass
    elif prediction_name == 'diabetes_prediction':
        diabetes_prediction()
    elif prediction_name == 'heart_disease_prediction':
        heart_disease_prediction()
    elif prediction_name == 'parkinsons_prediction':
        parkinsons_prediction()

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<footer>© 2023 Health Desk App. All rights reserved.</footer>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()