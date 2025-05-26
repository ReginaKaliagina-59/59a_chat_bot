import pandas as pd
import openai
import pickle
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel

openai.api_key = ""

# --- Config ---
CSV_PATH = "output_Monday_BI_data.csv"
# CSV_PATH = "output_Monday_BI_data_small.csv"
EMBED_PATH = "output_Monday_BI_data_full.pkl"
CHUNK_SIZE = 5  # number of rows per chunk
EMBED_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-3.5-turbo"

# --- Globals ---
chunks = []
embeddings = []

# --- Load CSV and create text chunks ---
def load_csv_chunks(file_path):
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    chunked = [
        df.iloc[i:i+CHUNK_SIZE].to_string(index=False)
        for i in range(0, len(df), CHUNK_SIZE)
    ]
    return chunked

def embed_chunks_bulk(text_chunks):
    result = []
    for i, chunk in enumerate(text_chunks):
        try:
            print(f"Embedding chunk {i+1}/{len(text_chunks)}...", end=" ")
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=chunk,
                timeout=15  # <-- prevent hanging
            )
            result.append(response.data[0].embedding)
            print("✅")
        except Exception as e:
            print(f"❌ Error: {e}")
            result.append([0.0] * 1536)  # dummy vector to preserve alignment
            time.sleep(1)  # optional cooldown
    return result

# --- Generate embeddings for each chunk ---
def embed_chunks(text_chunks):
    result = []
    count = 0
    total = len(text_chunks)
    for chunk in text_chunks:
        response = openai.embeddings.create(
            model=EMBED_MODEL,
            input=chunk
        )
        result.append(response.data[0].embedding)
        print(f'{count}/{total}')
        count += 1
    return result

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

# --- Main flow ---
def main():
    global chunks, embeddings

    # Load chunks and embeddings from disk
    # with open(EMBED_PATH, "rb") as f:
    #     chunks, embeddings = pickle.load(f)

    if (len(embeddings) == 0):
        print(f"Loading CSV from {CSV_PATH}...")
        chunks = load_csv_chunks(CSV_PATH)
        print(f"Split into {len(chunks)} chunks. Generating embeddings...")

        embeddings = embed_chunks_bulk(chunks)
        print("Embeddings complete.")

        # Save to disk
        with open(EMBED_PATH, "wb") as f:
            pickle.dump((chunks, embeddings), f)

    print("\nReady for questions. Type 'exit' to quit.\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        top_chunks = get_top_chunks(question)
        context = "\n\n".join(top_chunks)
        answer = ask_gpt(context, question)
        print("\nAnswer:", answer, "\n")

if __name__ == "__main__":
    main()