import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pickle

# App Title and Description
st.write("""
# RCA-GGBFS Concrete Compressive Strength Prediction
This app predicts the **Compressive Strength (Cs in MPa)** of RCA concrete incorporating GGBFS using mix design parameters.
""")
st.write('---')

# Optional image
image = Image.open('soil.jpg')  # Replace with a relevant image if needed
st.image(image, use_column_width=True)

# Load dataset
data = pd.read_csv("rca_ggbfs_concrete.csv")  # Replace with your actual file name

# Show dataset summary
st.subheader('Data Overview')
st.write(data.head())
st.write("Missing values:")
st.write(data.isna().sum())
st.write("Correlation matrix:")
st.write(data.corr())

# Sidebar input section
st.sidebar.header('Set Input Parameters')

def user_input():
    W_B = st.sidebar.slider('Water/Binder Ratio (W/B)', 0.25, 0.75, 0.479334)
    RA = st.sidebar.slider('Recycled Aggregate (%)', 0.0, 100.0, 61.158537)
    GGBFS = st.sidebar.slider('GGBFS (%)', 0.0, 90.0, 33.018293)
    Sp = st.sidebar.slider('Superplasticizer (kg)', 0.0, 7.8, 1.597024)
    Age = st.sidebar.slider('Age (days)', 7, 90, 34)

    features = pd.DataFrame({
        'W/B': [W_B],
        'RA%': [RA],
        'GGBFS%': [GGBFS],
        'Sp (kg)': [Sp],
        'Age (days)': [Age]
    })
    return features

# Get user input
df = user_input()

# Show user input
st.header('User Input Parameters')
st.write(df)
st.write('---')

# Load pre-trained model
model = pickle.load(open('optimized_gbrt_model.pkl', 'rb'))  # Make sure model is retrained for new input structure

# Predict compressive strength
st.header('Predicted Compressive Strength (Cs in MPa)')
prediction = model.predict(df)
st.write(round(prediction[0], 2))
st.write('---')
