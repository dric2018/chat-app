import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import os

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

import tomllib

from utils import start_metrics_server


with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)["tool"]["db"]

metrics = start_metrics_server()

# DB connection

# Handling PDF Upload & Processing
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
if uploaded_file:
    # Save temp file to load it
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # Split PDF into chunks
    loader = PyMuPDFLoader("temp.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"], 
        chunk_overlap=config["chunk_overlap"]
    )
    chunks = text_splitter.split_documents(docs)

    # Add to ChromaDB (Chroma handles default embedding internally)
    collection.add(
        documents=[c.page_content for c in chunks],
        ids=[f"id_{i}" for i in range(len(chunks))]
    )
    st.sidebar.success("PDF Indexed!")

# Enhanced Chat with Retrieval
if prompt := st.chat_input("Ask anything"):
    # SEARCH: Get the top 3 relevant chunks from the DB
    results = collection.query(query_texts=[prompt], n_results=3)
    context = "\n".join(results['documents'][0])
    
    # AUGMENT: Add that context to the prompt sent to vLLM
    augmented_prompt = f"Context from PDF: {context}\n\nQuestion: {prompt}"
    


st.title("📄 PDF Chat")

# --- UI: Sidebar History ---
st.sidebar.title("📚 Your Library")
existing_docs = collection.get()
if existing_docs['ids']:
    # Unique filenames/IDs stored in metadata
    st.sidebar.write(f"Indexed chunks: {len(existing_docs['ids'])}")
    if st.sidebar.button("Clear All History"):
        chroma_client.delete_collection("pdf_history")
        st.rerun()

# --- UI: Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Point to your Nginx proxy or vLLM container directly
client = OpenAI(base_url="http://localhost/v1", api_key="token-not-needed")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# PDF Upload (Integration point for RAG logic)
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

if prompt := st.chat_input("Ask about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Stream the response for that "live" feel
        stream = client.chat.completions.create(
            model=os.getenv("VLLM_MODEL", "model-name"),
            messages=st.session_state.messages,
            stream=True,
        )
        response = st.write_stream(stream)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
