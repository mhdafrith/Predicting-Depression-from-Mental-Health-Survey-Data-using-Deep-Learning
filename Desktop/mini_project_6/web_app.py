import streamlit as st
import torch
import torch.nn as nn
import numpy as np

import torch.nn as nn

 
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(14, 7)
        self.relu = nn.ReLU()
        self.output = nn.Linear(7, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x
model=NeuralNetwork()




# Load the trained model

def load_model():
    model = NeuralNetwork()
    state_dict = torch.load("Desktop/mini_project_6/model.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)  
    model.eval()
    return model

# Load the model once
model = load_model()


page=st.sidebar.selectbox('Page Navigation', ["Prediction","About"])
st.sidebar.markdown("""-----""")
st.sidebar.write('Created by [Mohamed Afrith](https://www.linkedin.com/in/mohamed-afrit-s/)')
if page=="Prediction":

    # Streamlit UI
    st.title("Depression Prediction App")
    st.write("Enter details below to predict depression.")

    # Input Fields
    gender = st.selectbox("Gender", ["Male", "Female"])
    working_status = st.selectbox("Are you a Working Professional or a Student?", ["Student", "Working Professional"])



    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    academic_pressure = st.slider("Academic Pressure", 0, 5, 0)
    work_pressure = st.slider("Work Pressure", 0, 5, 0)
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0)
    study_satisfaction = st.slider("Study Satisfaction", 0, 5, 0)
    job_satisfaction = st.slider("Job Satisfaction", 0, 5, 0)
    sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "6-7 hours", "7-8 hours", "More than 8 hours"])
    diet = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
    suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])
    study_hours = st.number_input("Work/Study Hours", min_value=1, max_value=16, value=8)
    financial_stress = st.slider("Financial Stress", 1, 5, 1)
    fam_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

    # Mapping categorical inputs to numerical values
    sleep_map = {"Less than 5 hours": 1, "5-6 hours": 2, "6-7 hours": 3, "7-8 hours": 4, "More than 8 hours": 5}
    diet_map = {"Healthy": 1, "Moderate": 2, "Unhealthy": 3}
    suicidal_map = {"Yes": 1, "No": 0}
    fam_history_map = {"Yes": 1, "No": 0}
    # Mapping categorical inputs to numerical values
    gender_map = {"Male": 0, "Female": 1, "Other": 2}
    working_status_map = {"Student": 0, "Working Professional": 1}

    import pickle

    # Load the saved scaler
    with open("Desktop/mini_project_6/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Create input array
    data = np.array([
        gender_map[gender], age, working_status_map[working_status],  
        academic_pressure, work_pressure, cgpa, study_satisfaction, job_satisfaction,
        sleep_map[sleep_duration], diet_map[diet], suicidal_map[suicidal_thoughts],
        study_hours, financial_stress, fam_history_map[fam_history]
    ], dtype=np.float32).reshape(1, -1)


    # Scale user input using the loaded scaler
    data_scaled = scaler.transform(data)

    # Convert to tensor
    data_tensor = torch.tensor(data_scaled, dtype=torch.float32)



    if st.button("Predict"):
        with torch.no_grad():
            prediction = model(data_tensor)
            prediction = prediction.item()
            result = "Depressed" if prediction > 0.5 else "Not Depressed"

            # Define Colors Based on Prediction
            color = "red" if result == "Depressed" else "green"

            # Display Result with Color Formatting
            st.markdown(
                f"<h2 style='color: {color}; text-align: center;'>Prediction: {result}</h2>",
                unsafe_allow_html=True
            )


# About Page
elif page == "About":
    st.title("Predicting Depression from Mental Health Survey Data using Deep Learning")
    st.image("Desktop/mini_project_6/premium_photo-1682689380897-446360437f25.jpg") # Add an image
    st.write(
        """
        ***About This Project:*** 
        **This project aims to predict depression using survey data and deep learning models.
        It analyzes various factors affecting mental health and helps in early detection.**
        
        - **Technologies Used**: Python, PyTorch, Streamlit, Scikit-learn
        - **Model**: Deep Learning-based classifier trained on mental health survey data
        - **Purpose**: To provide insights and awareness regarding mental health issues.
        """
    )
