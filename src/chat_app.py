from __init__ import logger, client
from config import CFG

import re
import sqlite3

import streamlit as st

from utils import retrieve_context, parse_llm_response

# Fix base URL for internal (docker) communication
client.base_url = f"http://vllm:{CFG.VLLM_PORT}/v1"

def parse_thinking_stream(stream):
    thinking_expander = st.expander("Show Reasoning", expanded=True)
    thinking_container = thinking_expander.empty()
    response_container = st.empty()

    full_thinking = ""
    full_response = ""
    is_thinking = False

    for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        
        if "<think>" in content:
            is_thinking = True
            content = content.replace("<think>", "")
            
        if "</think>" in content:
            is_thinking = False
            content = content.replace("</think>", "")

        if is_thinking:
            full_thinking += content
            thinking_container.markdown(full_thinking)
        else:
            full_response += content
            response_container.markdown(full_response)
            
    return full_thinking, full_response

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

def query_llm(
        input_text:str, 
        is_stream:bool=CFG.IS_STREAM
    ):

    st.session_state.messages.append(
        {"role": "user", "content": input_text}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    if is_stream:
        with st.chat_message("assistant"):
            # Stream the response for that "live" feel
            stream = client.chat.completions.create(
                model=CFG.BASE_MODEL,
                messages=st.session_state.messages,
                stream=True,
                temperature=CFG.GENERATION_TEMPERATURE,
                max_tokens=CFG.MAX_TOKENS
            )

            # response = st.write_stream(stream)
            thinking, final_sql = parse_thinking_stream(stream)
        
        st.session_state.messages.append({"role": "assistant", "content": final_sql})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing election data..."):
                response = client.chat.completions.create(
                    model=CFG.BASE_MODEL,
                    messages=st.session_state.messages,
                    stream=False,
                    temperature=CFG.GENERATION_TEMPERATURE,
                    max_tokens=CFG.MAX_TOKENS
                )
            
            raw_content = response.choices[0].message.content
            
            # Separate Thinking and SQL using your parser
            thinking, sql_query = parse_llm_response(raw_content)

            # Show thinking in a collapsible box
            if thinking:
                with st.expander("Show Reasoning"):
                    st.write(thinking)

            # Format the SQL
            st.code(sql_query, language="sql")

            st.session_state.messages.append({"role": "assistant", "content": raw_content})


st.title("📄 Chat CEI")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        
        # Check if this is an assistant message with thinking tags
        if message["role"] == "assistant" and "<think>" in content:
            # Parse the stored string
            match = re.search(r"<think>(.*?)</think>\s*(.*)", content, re.DOTALL)
            if match:
                think_text = match.group(1).strip()
                sql_text = match.group(2).strip()
                
                # Re-draw the UI elements
                with st.expander("Model Reasoning", expanded=False):
                    st.markdown(think_text)
                st.code(sql_text, language="sql")
            else:
                st.markdown(content)
        else:
            # Standard user message or fallback
            st.markdown(content)

if prompt := st.chat_input("Ask anything..."):
    query_llm(input_text=prompt)


    # Enhanced Chat with Retrieval
    # if prompt := st.chat_input("Ask anything"):
    #     response = chat_with_rag(prompt)
