from __init__ import logger, client
from config import CFG

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sqlite3

import streamlit as st

import tomllib

from utils import retrieve_context


def chat_with_rag(
        user_input:str, 
        db_conn:sqlite3.Connection
    ):
    
    # R: Retrieval
    context = retrieve_context(db_conn, user_input)
    
    # A: Augmentation (Chat Template)
    prompt = f"""Use the following context to answer the user's request.
    Context:
    {context}\n
    Request: {user_input}
    Answer:"""

    # G: Generation via vLLM
    response = client.chat.completions.create(
        model=CFG.BASE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=CFG.GENERATION_TEMPERATURE
    )
    
    return response.choices[0].message.content

# with open("pyproject.toml", "rb") as f:
#     config = tomllib.load(f)["tool"]["db"]

st.title("📄 PDF Chat")

# --- UI: Sidebar History ---
st.sidebar.title("📚 Your Library")
# existing_docs = collection.get()
# if existing_docs['ids']:
#     # Unique filenames/IDs stored in metadata
#     st.sidebar.write(f"Indexed chunks: {len(existing_docs['ids'])}")
#     if st.sidebar.button("Clear All History"):
#         chroma_client.delete_collection("pdf_history")
#         st.rerun()

# --- UI: Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# Enhanced Chat with Retrieval
# if prompt := st.chat_input("Ask anything"):
#     response = chat_with_rag(prompt)

    with st.chat_message("assistant"):
        # Stream the response for that "live" feel
        stream = client.chat.completions.create(
            model=CFG.BASE_MODEL,
            messages=st.session_state.messages,
            stream=True,
            temperature=CFG.GENERATION_TEMPERATURE
        )
        response = st.write_stream(stream)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
