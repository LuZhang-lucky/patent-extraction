import requests
import json

url = "http://192.168.10.236:11434/api/chat"

payload = {
    "model": "qwen3:8b",
    "messages": [
        {
            "role": "user",
            "content": "Reply only with: connection works"
        }
    ],
    "stream": False
}

response = requests.post(url, json=payload, timeout=120)
response.raise_for_status()

print(json.dumps(response.json(), ensure_ascii=False, indent=2))