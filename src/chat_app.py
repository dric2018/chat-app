from __init__ import logger
from config import CFG

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import plotly.express as px

import re

import streamlit as st
from agent import QueryIntent, HybridAgent

from utils import parse_llm_response

from prometheus_client import start_http_server


try:
    start_http_server(port=8001, addr='http://streamlit-app')    
except OSError:
    pass 

agent = HybridAgent(vllm_url=f"http://vllm:{CFG.VLLM_PORT}/v1")

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

def render_agent_response(response):
    """
    Renders the built-in <think> monologue, the investigation steps, 
    and the final data/interpretation.
    """
    if not isinstance(response, dict):
        response = response.model_dump()
    # HANDLE BUILT-IN MODEL THINKING (<think> tags)
    # The 'interpretation' or 'content' usually contains the raw LLM string
    raw_text = response.get("interpretation") or response.get("content", "")
    
    if isinstance(raw_text, str) and "<think>" in raw_text:
        match = re.search(r"<think>(.*?)</think>\s*(.*)", raw_text, re.DOTALL)
        if match:
            think_text = match.group(1).strip()
            final_text = match.group(2).strip()
            
            # with st.expander("💭 Model Internal Monologue", expanded=False):
            st.markdown(think_text)
            
            # Show the "cleaned" interpretation without the tags
            st.markdown(final_text)
        else:
            st.markdown(raw_text)
    else:
        # If no think tags, just show the text
        st.markdown(raw_text)

    # HANDLE AGENT INVESTIGATION STEPS (Tool Reasoning)
    if "steps" in response and response["steps"]:
        with st.expander("🔍 Investigation Path (Tools Used)", expanded=False):
            for i, step in enumerate(response["steps"]):
                st.markdown(f"**Step {i+1}:** {step}")

    # HANDLE DATA & VISUALS
    if response["type"] == "data":
        if "sql" in response:
            with st.expander("💻 Generated SQL Query"):
                st.code(response["sql"], language="sql")
        
        if response.get("intent") == QueryIntent.CHART:
            df = response["data"]
            fig = px.bar(df, x=df.columns[0], y=df.columns[1], title="Election Insights")
            st.plotly_chart(fig, use_container_width=True)
            
        with st.expander("📊 View Raw Data"):
            st.dataframe(response["data"])

def query_llm(input_text: str):
    final_answer = None

    # User Message
    with st.chat_message("user"):
        st.markdown(input_text)
        st.session_state.messages.append(HumanMessage(content=input_text))

    # Assistant Message
    with st.chat_message("assistant"):
        with st.status("🔍 Election Agent is thinking...", expanded=True) as status:
            for update in agent.get_answer(input_text):
                if update["type"] == "status":
                    status.write(f"⚙️ {update['content']}")
                
                elif update["type"] in ["text", "data", "final_sql", "final"]:
                    final_answer = update
                    status.update(label="✅ Processing Complete", state="complete", expanded=False)
                
                elif update["type"] == "error":
                    status.update(label="❌ Error", state="error")
                    st.error(update["content"])
                    return

        if final_answer:
            render_agent_response(final_answer)
            
            content_to_save = final_answer.get("content") or final_answer.get("interpretation", "")
            st.session_state.messages.append(AIMessage(content=content_to_save))
            print("ST messages: ", st.session_state.messages)

st.title("📄 Chat App")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    role = "user" if message.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(role)
        if role == "assistant":
            render_agent_response(message)
        else:
            st.markdown(message.content)

if prompt := st.chat_input("Ask anything..."):
    query_llm(input_text=prompt)

# if __name__=="__main__":
#     sys_prompt = """Answer the user in a casual manner"""
#     user_prompt = input(">> ")
    
#     messages = [
#         SystemMessage(content=sys_prompt), 
#         HumanMessage(content=user_prompt)
#     ]

#     while user_prompt !="exit":
#         response = agent.llm_with_tools.invoke(messages)
#         print(f"<< {response.content}\n")

#         messages.append(AIMessage(content=response.content))
#         user_prompt = input(">> ")
#         messages.append(HumanMessage(content=user_prompt))
