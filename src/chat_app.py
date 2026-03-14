from __init__ import logger
from config import CFG

import json

from langchain_core.messages import HumanMessage, AIMessage

import plotly.express as px

import re

import streamlit as st
from agent import QueryIntent, HybridAgent

from utils import parse_llm_response

from time import time

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
    if hasattr(response, "content"):
        response = response.content

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
    if CFG.DEBUG_MODE:
        if "steps" in response and response["steps"]:
            with st.expander("🔍 Investigation Path (Tools used)", expanded=False):
                for i, step in enumerate(response["steps"]):
                    st.markdown(f"**Step {i+1}:** {step}")

    # HANDLE DATA & VISUALS
    if response["type"] == "data":
        intent = response.get("intent")

        with st.expander("💻 Generated SQL Query"):
            st.code(response["final_sql"], language="sql")
    
        if intent.value == QueryIntent.CHART.value:
            df = response["data"]
            with st.expander("🖼️ Visualization "):
                fig = px.bar(df, x=df.columns[0], y=df.columns[1], title="Election Insights")
                st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📊 View Raw Data"):
            st.dataframe(response["data"])

def query_llm(input_text: str):

    current_history = st.session_state.get("messages", [])

    final_answer = None

    # User Message
    with st.chat_message("user"):
        st.markdown(input_text)
        st.session_state.messages.append(HumanMessage(content=input_text))

    # Assistant Message
    with st.chat_message("assistant"):
        with st.status("🔍 Thinking...", expanded=True) as status:
            start = time()
            for update in agent.get_answer(input_text, chat_history=current_history):
                if update["type"] == "status":
                    new_label = f"{update['content']}"
                    status.update(label=new_label, state="running")
                    status.write(f"⚙️ {new_label}")

                    if "reasoning" in update:
                        st.info(f"**Reasoning:** {update['reasoning']}")
                    
                    if "possible_clarification" in update:
                        st.warning(f"**Possible Clarification Question:** {update['possible_clarification']}")
                            
                elif update["type"] in ["text", "data", "final_sql", "final"]:                    
                    final_answer = update
                    duration = (time() - start) /60

                    status.update(
                        label=f"✅ Processing Complete (in {duration:.3f} min)", 
                        state="complete", 
                        expanded=False
                    )
                    
                elif update["type"] == "error":
                    status.update(label="❌ Error", state="error")
                    st.error(update["content"])

        if final_answer:
            render_agent_response(final_answer)

            # json_content = json.dumps(final_answer)

            # st.session_state.messages.append(
            #     AIMessage(content=json_content)
            # )
        
            st.session_state.messages.append(
                AIMessage(
                    content=final_answer.get("content", ""), 
                    additional_kwargs={"full_response": final_answer}
                )
            )

def select_suggestion():
    if st.session_state.suggestion_box:
        st.session_state.chat_input_key = st.session_state.suggestion_box

st.title("📄🇨🇮 CIV Election Master")
st.markdown(
    "Hi! I can answer questions about the 2025 Legislative elections in Côte d'Ivoire. " \
    "You can ask general, ranking, aggregation questions etc.")

# Example suggestions
rag_examples = [
    "What was the final voter turnout percentage in the 2025?",
    "Summarize the election results for the Tiapoum constituency.",
    "Which party saw the biggest increase in seat share compared to the last cycle?",
    "How does the urban vs. rural turnout compare in the latest results?",
]
SUGGESTIONS = [
    "What is the total votes by party ?", 
    "Which region has the most voters?", # ✅
    "What is the turnout in the region with the most voters?", # ✅  #same as previous query but kept for tests
    "Which candidates won the elections in Abidjan?" , # ✅ 
    "How many seats did ADCI win?", # ✅ 
    "Who won the elections in tiapum",
    "Top 10 candidates by score in region Nawa.", # ✅ 
    "Participation rate by region", # ✅ 
    "Distribution of winners per party", # ✅ 
    "Which party did win the most seats?", # ✅ 
    "Show me the distribution of voters per region", # ✅ 
    "histogram of the number of candidates per party and per region"
] + rag_examples

# Show only before the first message
if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = []

selected_option = st.pills(
    label="Examples:", 
    options=SUGGESTIONS,
    key="suggestion_box",
    on_change=select_suggestion,
    selection_mode="single"
)

# Display chat history
# for message in st.session_state.messages:
#     role = "user" if message.type == "human" else "assistant"
    
#     with st.chat_message(role):
#         if role == "assistant":
#             full_data = message.additional_kwargs.get("full_response")
#             if full_data:
#                 render_agent_response(full_data)            
#         else:
#             st.markdown(message.content)

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            # Pull the full dictionary back out of the metadata
            full_response = message.additional_kwargs.get("full_response")
            if full_response:
                render_agent_response(full_response)
            else:
                st.markdown(message.content)


chat_prompt = st.chat_input("Ask anything...", key="chat_input_key")


if chat_prompt:
    query_llm(input_text=chat_prompt)

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
