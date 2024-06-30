import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# Define the neural network model
hidden1_size = 16
hidden2_size = 8
hidden3_size = 4
output_size = 1

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, hidden3_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden1_size)
        self.layer2 = nn.Linear(hidden1_size, hidden2_size)
        self.layer3 = nn.Linear(hidden2_size, hidden3_size)
        self.output = nn.Linear(hidden3_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

# Load the trained model
input_size = 8
my_model = NeuralNetwork(input_size, hidden1_size, hidden2_size, hidden3_size, output_size)
my_model.load_state_dict(torch.load('my_model.pth'))
my_model.eval()  # Set the model to evaluation mode

# Function to make predictions
def predict_diabetes(input_data):
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad():
        output = my_model(input_tensor)
    return output.item()

# Streamlit app
st.title("Simple Web App to Predict Risk of Diabetes")

# Input prompts for user to enter data
st.header("Enter the patient data:")
input_data = []

input_data.append(st.slider('Number of pregnancies', min_value=0, max_value=20, value=0))
input_data.append(st.number_input('Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test.', value=0.0))
input_data.append(st.number_input('Blood Pressure (mmHg)', value=0.0))
input_data.append(st.number_input('Skin Thickness: Triceps Skinfold Thickness (mm)', value=0.0))
input_data.append(st.number_input('Insulin', value=0.0))
input_data.append(st.number_input('BMI', value=0.0))
input_data.append(st.number_input('Diabetes Pedigree Function: A function that scores likelihood of diabetes based on family history', value=0.0))
input_data.append(st.number_input('Age', value=0.0))

# Predict button
if st.button("Predict"):
    input_data = np.array(input_data).reshape(1, -1)

    prediction = predict_diabetes(input_data[0])
    st.write(f"Prediction probability: {prediction:.2f}")
    if prediction >= 0.5:  # Useing a threshold of 0.5 for classification
        st.write("The model predicts that the patient has a high risk of diabetes.")
    else:
        st.write("The model predicts that the patient has a low risk of diabetes.")
