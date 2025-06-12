import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
from pytorch_model import AutoFeatureGenerator  # Import the PyTorch model
from rag_ai.eda_agent import EdaAgent
# Constants
UPLOAD_FILE_TYPES = ["pdf", "csv"]
EDA_AGENT = EdaAgent()  # Initialize the EDA agent

# Streamlit app
st.title("Data Insights and Visualization")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF or CSV file", type=UPLOAD_FILE_TYPES)

if uploaded_file is not None:
    # Read the uploaded file
    if uploaded_file.type == "application/pdf":
        # Use a library like PyPDF2 to read the PDF file
        import PyPDF2
        pdf_file = PyPDF2.PdfFileReader(uploaded_file)
        text = ""
        for page in pdf_file.pages:
            text += page.extractText()
        df = pd.DataFrame([text])
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)

    # Preprocess the data
    df = EDA_AGENT.preprocess_data(df)

    # Generate insights and visualizations
    insights = EDA_AGENT.generate_insights(df)
    visualizations = EDA_AGENT.generate_visualizations(df)

    # Display the insights and visualizations
    st.write(insights)
    st.write(visualizations)

    # Use the PyTorch model to generate automatic features
    auto_features = AutoFeatureGenerator(df)
    st.write(auto_features)