import os
import time
import faiss
import numpy as np
import streamlit as st
import requests
from bs4 import BeautifulSoup
from mistralai import Mistral, UserMessage

# Set API Key
os.environ["MISTRAL_API_KEY"] = "pSnb6dOIGJqlPqhVuNo9nxC02ilfYPls"
api_key = os.getenv("MISTRAL_API_KEY")

# Check API Key
if not api_key:
    st.error("API Key is missing! Set your MISTRAL_API_KEY.")
    st.stop()

# Fetch Policies
def get_policies():
    url = "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    raw_policies = [tag.text.strip() for tag in soup.find_all("div") if tag.text.strip()]
    
    # Clean policies: Remove excessive newlines and spaces
    cleaned_policies = [" ".join(policy.split()) for policy in raw_policies]
    
    return cleaned_policies[:10]  # Limit to 10 policies

# Fetch policies and chunk text
policies = get_policies()
policy_titles = [f"Policy {i+1}: {policies[i][:50]}..." for i in range(len(policies))]  # Display only first 50 chars

chunks = [chunk for policy in policies for chunk in policy.split(". ")]

# Generate Embeddings with API Rate Limit Handling
def get_embeddings(chunks, batch_size=1, delay=3, max_retries=5):
    client = Mistral(api_key=api_key)
    embeddings = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        retries = 0

        while retries < max_retries:
            try:
                response = client.embeddings.create(model="mistral-embed", inputs=batch)
                embeddings.extend([e.embedding for e in response.data])
                time.sleep(delay)  # Add delay to prevent hitting rate limits
                break  # Exit retry loop if successful
            except Exception as e:
                print(f"API Error: {e}. Retrying in {delay * 2} seconds...")
                time.sleep(delay * 2)
                retries += 1
        
        if retries == max_retries:
            print("Max retries reached. Skipping batch.")

    return embeddings

text_embeddings = get_embeddings(chunks)
d = len(text_embeddings[0])  # Embedding dimension
index = faiss.IndexFlatL2(d)
index.add(np.array(text_embeddings))

# ------------------ Streamlit UI ------------------

st.title("UDST Policy Chatbot")

# Dropdown to select a policy
selected_policy = st.selectbox("Select a UDST Policy:", policy_titles)

# Query input box
query = st.text_input("Enter your question:")

if query:
    client = Mistral(api_key=api_key)
    query_embedding = np.array(client.embeddings.create(model="mistral-embed", inputs=[query]).data[0].embedding).reshape(1, -1)
    D, I = index.search(query_embedding, k=2)
    retrieved_chunks = [chunks[i] for i in I[0]]
    
    # Create prompt for chatbot
    prompt = f"""
    Context:
    {' '.join(retrieved_chunks)}
    Query: {query}
    Answer:
    """

    response = client.chat.complete(model="mistral-large-latest", messages=[UserMessage(content=prompt)])
    
    # Display selected policy info
    st.write(f"**You selected:** {selected_policy}")
    st.write("### Answer:")
    st.write(response.choices[0].message.content)
