import os
import time
import numpy as np
import streamlit as st
import requests
import faiss
from bs4 import BeautifulSoup
from mistralai import Mistral, UserMessage

# Set API Key for Mistral
os.environ["MISTRAL_API_KEY"] = "pSnb6dOIGJqlPqhVuNo9nxC02ilfYPls"
api_key = os.getenv("MISTRAL_API_KEY")

# Ensure API key is set
if not api_key:
    st.error("🚨 API Key is missing! Set your MISTRAL_API_KEY.")
    st.stop()

# Initialize Mistral Client
client = Mistral(api_key=api_key)

# Function to scrape UDST policies
def get_policies():
    url = "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures"
    response = requests.get(url)

    if response.status_code != 200:
        raise ValueError("Failed to fetch UDST policies. Check the website status.")

    soup = BeautifulSoup(response.text, "html.parser")
    raw_policies = [tag.text.strip() for tag in soup.find_all("div") if tag.text.strip()]

    # Clean policies: remove unnecessary newlines and whitespace
    cleaned_policies = [" ".join(policy.split()) for policy in raw_policies]
    
    return cleaned_policies[:10] if len(cleaned_policies) >= 10 else cleaned_policies

# Fetch policies and ensure at least 10 are available
policies = get_policies()
if len(policies) < 10:
    st.warning("⚠️ Less than 10 policies found. Adding placeholders.")
    policies += [f"Placeholder Policy {i+1}" for i in range(10 - len(policies))]

# Function to chunk policies for embeddings
def chunk_text(text, chunk_size=256):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Process policies into chunks
chunks = [chunk for policy in policies for chunk in chunk_text(policy)]

# Function to get embeddings
def get_embeddings(chunks, batch_size=1, delay=3, max_retries=5):
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        retries = 0
        while retries < max_retries:
            try:
                response = client.embeddings.create(model="mistral-embed", inputs=batch)
                embeddings.extend([e.embedding for e in response.data])
                time.sleep(delay)
                break
            except Exception as e:
                st.warning(f"⚠️ API Error: {e}. Retrying in {delay * 2} seconds...")
                time.sleep(delay * 2)
                retries += 1
        if retries == max_retries:
            st.error("🚨 Max retries reached. Skipping batch.")
    return embeddings

# Generate embeddings
text_embeddings = get_embeddings(chunks)

# Store embeddings in FAISS
d = len(text_embeddings[0]) if text_embeddings else 0
index = faiss.IndexFlatL2(d)
index.add(np.array(text_embeddings))

# Streamlit UI Setup
st.title("📜 UDST Policy Chatbot 🤖")
st.write("Ask about **UDST policies** and get instant answers! 🎓")

# Dropdown for policy selection
selected_policy = st.selectbox(
    "📜 Select a UDST Policy:", [f"Policy {i+1}: {policies[i]}" for i in range(len(policies))]
)

# Text input for query
query = st.text_input("💬 Enter your question:")

# Function to get query embedding
def get_query_embedding(query):
    response = client.embeddings.create(model="mistral-embed", inputs=[query])
    return np.array(response.data[0].embedding)

# Function to retrieve and answer questions
def ask_mistral(prompt):
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[UserMessage(content=prompt)]
    )
    return response.choices[0].message.content

# Handle query
if query:
    query_embedding = get_query_embedding(query).reshape(1, -1)
    D, I = index.search(query_embedding, k=2)
    retrieved_chunks = [chunks[i] for i in I[0]]

    # Construct prompt
    prompt = f"""
    Context:
    {' '.join(retrieved_chunks)}
    Query: {query}
    Answer:
    """

    # Display selected policy
    st.markdown(f"📌 **You selected:**  📜 {selected_policy}")

    # Generate answer
    answer = ask_mistral(prompt)

    # Display answer
    st.markdown("✅ **Answer:**")
    st.write(answer)

# Footer
st.markdown("---")
st.markdown("🤖 **Built with Mistral AI, FAISS & Streamlit** | 🚀 **By Noor Zena**")
