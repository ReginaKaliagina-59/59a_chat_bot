import streamlit as st
import pandas as pd
import openai
import os
import platform
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# --- Find most relevant chunks ---
def get_top_chunks(query, top_k=3):
    query_embed = openai.embeddings.create(
        model=EMBED_MODEL,
        input=query
    ).data[0].embedding

    similarities = cosine_similarity([query_embed], embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# --- Ask GPT ---
def ask_gpt(context, question):
    prompt = f"Here is some data:\n\n{context}\n\nQuestion: {question}\nAnswer:"
    response = openai.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300
    )
    return response.choices[0].message.content

# 🔍 Detect the OS type
os_type = platform.system()

# Securely initialize OpenAI API Key using environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Before running your app, set the API key in your terminal:
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

# Title of the app
st.title("Chatbot")

# Load the dataset directly
dataset_path = "output_Monday_BI_data.csv"  # Provide full path if not in the same folder

EMBED_PATH = "output_Monday_BI_data_full.pkl"
CHUNK_SIZE = 5  # number of rows per chunk
EMBED_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-3.5-turbo"

chunks = []
embeddings = []

try:
    # df = pd.read_csv(dataset_path, encoding='ISO-8859-1')
    # Load chunks and embeddings from disk
    with open(EMBED_PATH, "rb") as f:
        chunks, embeddings = pickle.load(f)

    st.success("Dataset Loaded Successfully!")
    # st.dataframe(df.head())  # Display the first 5 rows of the dataset
except FileNotFoundError:
    st.error(f"Dataset not found at path: {EMBED_PATH}")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while reading the file: {e}")
    st.stop()

# User input
question = st.text_input("Ask a question about your dataset:")

if question:
    try:
        top_chunks = get_top_chunks(question)
        context = "\n\n".join(top_chunks)
        answer = ask_gpt(context, question)

        st.write("**Bot:**", answer)

    except Exception as e:
        st.error(f"An error occurred during OpenAI request: {e}")
