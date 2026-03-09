#/bin/bash

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-1.7B",
    "messages": [
    {"role": "system", "content": "You are a casual assistant, reply with warm and respectful answers"},
    {"role": "user", "content": "Hello"}
    ]
  }'
