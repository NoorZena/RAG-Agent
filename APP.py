import os
import time
import faiss
import numpy as np
import streamlit as st
import requests
from bs4 import BeautifulSoup
from mistralai import Mistral, UserMessage

# 🎯 Set API Key
os.environ["MISTRAL_API_KEY"] = "pSnb6dOIGJqlPqhVuNo9nxC02ilfYPls"
api_key = os.getenv("MISTRAL_API_KEY")

# 🚨 Check if API Key is missing
if not api_key:
    st.error("❌ API Key is missing! Please set your `MISTRAL_API_KEY`.")
    st.stop()

# 📜 Function to fetch & clean UDST policies
def get_policies():
    """
    Fetches UDST policies from the official website.
    Extracts only meaningful policy titles and ensures at least 10 policies.
    """
    url = "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures"
    response = requests.get(url)

    if response.status_code != 200:
        st.error("⚠️ Failed to fetch UDST policies. Check the website status.")
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract policy titles from <h2> and <h3> headings
    raw_policies = [tag.text.strip() for tag in soup.find_all(["h2", "h3"])]

    # Remove duplicates and extra spaces
    cleaned_policies = list(set([" ".join(policy.split()) for policy in raw_policies]))

    # Ensure at least 10 policies (fallback if fewer are found)
    while len(cleaned_policies) < 10:
        cleaned_policies.append(f"Placeholder Policy {len(cleaned_policies)+1}")

    return cleaned_policies[:10]  # Ensure exactly 10 policies

# 📥 Fetch policies and prepare dropdown options
policies = get_policies()

if policies:
    policy_titles = [f"📜 Policy {i+1}: {policies[i]}" for i in range(len(policies))]
else:
    policy_titles = ["❌ No policies found. Please check extraction."]

# ✂️ Chunk the policy text for processing
def chunk_text(text, chunk_size=256):
    """
    Splits text into smaller chunks for efficient processing.
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# 📦 Process all policies into chunks
chunks = [chunk for policy in policies for chunk in chunk_text(policy)]

# 🧠 Generate Embeddings with API Rate Limit Handling
def get_embeddings(chunks, batch_size=1, delay=3, max_retries=5):
    """
    Generates embeddings for text chunks using Mistral AI.
    Uses batch processing with retry logic to avoid rate limits.
    """
    client = Mistral(api_key=api_key)
    embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        retries = 0

        while retries < max_retries:
            try:
                response = client.embeddings.create(model="mistral-embed", inputs=batch)
                embeddings.extend([e.embedding for e in response.data])
                time.sleep(delay)  # Delay to prevent hitting rate limits
                break  # Exit retry loop if successful
            except Exception as e:
                print(f"⚠️ API Error: {e}. Retrying in {delay * 2} seconds...")
                time.sleep(delay * 2)
                retries += 1

        if retries == max_retries:
            print("⏳ Max retries reached. Skipping batch.")

    return embeddings

# 🧠 Generate embeddings for policies
text_embeddings = get_embeddings(chunks)

# 🔍 Store Embeddings in FAISS Vector Database
d = len(text_embeddings[0])  # Embedding dimension
index = faiss.IndexFlatL2(d)
index.add(np.array(text_embeddings))

# 🎨 Streamlit UI
st.title("📜 UDST Policy Chatbot 🤖")
st.markdown("Ask about **UDST policies** and get instant answers! 🎓")

# 📌 Dropdown list to select a policy
selected_policy = st.selectbox("📜 Select a UDST Policy:", policy_titles)

# ✏️ Query input box
query = st.text_input("💬 Enter your question:")

# 🔎 Process Query when user enters text
if query:
    client = Mistral(api_key=api_key)
    
    # 🧠 Generate embedding for the query
    query_embedding = np.array(client.embeddings.create(model="mistral-embed", inputs=[query]).data[0].embedding).reshape(1, -1)

    # 🔍 Retrieve most relevant chunks
    D, I = index.search(query_embedding, k=2)
    retrieved_chunks = [chunks[i] for i in I[0]]

    # 📝 Create prompt for chatbot
    prompt = f"""
    Context:
    {' '.join(retrieved_chunks)}
    Query: {query}
    Answer:
    """

    # 🗣️ Ask Mistral AI
    response = client.chat.complete(model="mistral-large-latest", messages=[UserMessage(content=prompt)])

    # ✅ Display selected policy and chatbot response
    st.write(f"📌 **You selected:** {selected_policy}")
    st.markdown("### ✅ Answer:")
    st.write(response.choices[0].message.content)

# 🎉 Footer
st.markdown("---")
st.markdown("🤖 **Built with Mistral AI, FAISS & Streamlit** | 🚀 **By Noor Zena**")
