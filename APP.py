import os
import streamlit as st
from mistralai import Mistral, UserMessage

# Set API Key for Mistral
os.environ["MISTRAL_API_KEY"] = "pSnb6dOIGJqlPqhVuNo9nxC02ilfYPls"
api_key = os.getenv("MISTRAL_API_KEY")

# Ensure API key is set
if not api_key:
    st.error("🚨 API Key is missing! Set your MISTRAL_API_KEY environment variable.")
    st.stop()

# Initialize Mistral Client
client = Mistral(api_key=api_key)

# List of UDST policies
policies = [
    {"title": "Academic Annual Leave Policy - V1", "number": "PL-AC-06"},
    {"title": "Academic Appraisal Policy - V1", "number": "PL-AC-09"},
    {"title": "Academic Credentials Policy - V1", "number": "PL-AC-02"},
    {"title": "Academic Freedom Policy - V1", "number": "PL-AC-10"},
    {"title": "Academic Members' Retention Policy - V1", "number": "PL-AC-12"},
    {"title": "Academic Professional Development Policy - V1", "number": "PL-AC-11"},
    {"title": "Academic Qualifications Policy - V2", "number": "PL-AC-03"},
    {"title": "Credit Hour Policy - V1", "number": "PL-AC-26"},
    {"title": "Intellectual Property Policy - V1", "number": "PL-AC-14"},
    {"title": "Joint Appointment Policy - V1", "number": "PL-AC-20"},
]

# Streamlit UI Setup
st.title("📜 UDST Policy Chatbot 🤖")
st.write("Ask about **UDST policies** and get instant answers! 🎓")

# Dropdown for policy selection
selected_policy = st.selectbox(
    "📜 Select a UDST Policy:",
    [f"{policy['title']} ({policy['number']})" for policy in policies]
)

# Text input for query
query = st.text_input("💬 Enter your question:")

# Function to retrieve and answer questions
def ask_mistral(prompt):
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[UserMessage(content=prompt)]
    )
    return response.choices[0].message.content

# Handle query
if query:
    # Construct prompt
    prompt = f"""
    You are an expert on university policies. Provide detailed information on the following policy:

    Policy: {selected_policy}

    Question: {query}

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
st.markdown("🤖 **Built with Mistral AI & Streamlit** | 🚀 **By Noor Zena**")
