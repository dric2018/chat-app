from __init__ import logger, client
from config import CFG
from utils import parse_llm_response

import time


if __name__=="__main__":
    user_q = "Explain how a CPU runs an LLM in one sentence. And tell us why a GPU matters Answer in French"

    msg = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_q}
        ]

    start_time = time.perf_counter()

    response = client.chat.completions.create(
        model=CFG.BASE_MODEL,
        messages=msg,
        temperature=CFG.GENERATION_TEMPERATURE,
        max_tokens=CFG.MAX_TOKENS
    )

    raw_out = response.choices[0].message.content

    thinking, resp = parse_llm_response(raw_out)

    end_time = time.perf_counter()
    duration = end_time - start_time

    print(f"\nUser: {user_q}\n")
    print(f"Assistant:\nThinking: {thinking}\nResponse: {resp}\n")

    logger.info(f"Request took {duration:.2f} seconds")
