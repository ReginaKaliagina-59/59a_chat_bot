import streamlit as st
import pandas as pd
import openai
import os
import platform

# 🔍 Detect the OS type
os_type = platform.system()

# ✅ Securely initialize OpenAI API Key using environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ Before running your app, set the API key in your terminal:
# export OPENAI_API_KEY='sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

if not openai.api_key:
    st.warning("OpenAI API key not found. Please enter it to continue.")
    api_key_input = st.text_input("Enter your OpenAI API Key:", type="password")
    
    if api_key_input:
        openai.api_key = api_key_input
        
        # If Windows, set the environment variable for this session
        if os_type == "Windows":
            os.system(f'setx OPENAI_API_KEY "{api_key_input}"')
            st.success("API Key successfully set for this session!")
        else:
            os.environ["OPENAI_API_KEY"] = api_key_input
            st.success("API Key successfully set for this session!")

    if not openai.api_key:
        st.error("OpenAI API key not found. Application cannot proceed.")
        st.stop()

# ✅ Title of the app
st.title("Chatbot with Dataset Analysis")

# ✅ Load the dataset directly
dataset_path = "output_Monday_BI_data.csv"  # Provide full path if not in the same folder

try:
    df = pd.read_csv(dataset_path, encoding='ISO-8859-1')
    st.success("Dataset Loaded Successfully!")
    st.dataframe(df.head())  # Display the first 5 rows of the dataset
except FileNotFoundError:
    st.error(f"Dataset not found at path: {dataset_path}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while reading the file: {e}")
    st.stop()


# ✅ Define the OpenAI Interaction Logic
def ask_openai(question, df_sample):
    """
    Sends a question to the OpenAI model and provides a sample of the dataset for context.
    """
    system_message = {
        "role": "system",
        "content": "You are a data analyst assistant. You will analyze datasets and provide insights based on user questions."
    }
    
    user_message = {
        "role": "user",
        "content": f"Here is a sample of the dataset:\n\n{df_sample.head(5).to_string()}\n\nMy question is: {question}"
    }

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[system_message, user_message],
            temperature=0.5
        )
        answer = response['choices'][0]['message']['content']
        return answer
    except Exception as e:
        st.error(f"An error occurred during OpenAI request: {e}")
        return None


# ✅ User input and bot response
user_input = st.text_input("Ask a question about your dataset:")

if user_input:
    with st.spinner("Generating response..."):
        response = ask_openai(user_input, df)
        if response:
            st.write("**Bot:**", response)
        else:
            st.error("Failed to retrieve a response from OpenAI.")
