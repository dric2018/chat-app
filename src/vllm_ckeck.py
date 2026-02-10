from __init__ import logger, client
from config import CFG

msg = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain how a CPU runs an LLM in one sentence."}
    ]

response = client.chat.completions.create(
    model=CFG.BASE_MODEL,
    messages=msg,
    temperature=CFG.GENERATION_TEMPERATURE,
    max_tokens=CFG.MAX_TOKENS
)

logger.info(f"Assistant: {response.choices[0].message.content}")

# prompt = tokenizer.apply_chat_template(
#     messages, 
#     tokenize=False, 
#     add_generation_prompt=True, 
# )

# response = client.completions.create(
#     model=CFG.BASE_MODEL,
#     prompt=prompt,
#     max_tokens=100
# )

# print(response.choices[0].text)
