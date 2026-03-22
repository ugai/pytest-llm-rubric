"""Experiment: qwen3.5 via native Ollama API + OpenAI raw response."""

import sys

import httpx

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.3.2:11434"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.5:9b"

SYSTEM = (
    'You are a rubric grader. You will be given a DOCUMENT and a CRITERION.\n'
    'Determine whether the document expresses the criterion.\n'
    'Respond with a single word: "PASS" or "FAIL".\n'
    'Your response must be exactly one word. Do not explain.'
)

DOC = "Agents must prioritize bug issues over enhancement issues."
CRITERION = "Bug issues are prioritized over enhancement issues."
USER_MSG = f"DOCUMENT:\n{DOC}\n\nCRITERION:\n{CRITERION}"

client = httpx.Client(timeout=120.0)

# --- Test 1: Native Ollama API (think=true) ---
print("=== Native Ollama API (think=true) ===")
resp = client.post(f"{BASE_URL}/api/chat", json={
    "model": MODEL,
    "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_MSG},
    ],
    "think": True,
    "stream": False,
})
data = resp.json()
msg = data.get("message", {})
print(f"  content:  {msg.get('content')!r}")
print(f"  thinking: {str(msg.get('thinking', ''))[:200]!r}")
print(f"  role:     {msg.get('role')!r}")

# --- Test 2: Native Ollama API (think=false) ---
print("\n=== Native Ollama API (think=false) ===")
resp = client.post(f"{BASE_URL}/api/chat", json={
    "model": MODEL,
    "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_MSG},
    ],
    "think": False,
    "stream": False,
})
data = resp.json()
msg = data.get("message", {})
print(f"  content:  {msg.get('content')!r}")
print(f"  thinking: {str(msg.get('thinking', ''))[:200]!r}")

# --- Test 3: Native Ollama API (/nothink in system prompt) ---
print("\n=== Native Ollama API (/nothink in system) ===")
resp = client.post(f"{BASE_URL}/api/chat", json={
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "/nothink\n" + SYSTEM},
        {"role": "user", "content": USER_MSG},
    ],
    "stream": False,
})
data = resp.json()
msg = data.get("message", {})
print(f"  content:  {msg.get('content')!r}")
print(f"  thinking: {str(msg.get('thinking', ''))[:200]!r}")

# --- Test 4: OpenAI-compat raw JSON ---
print("\n=== OpenAI-compat /v1/chat/completions (raw JSON) ===")
resp = client.post(f"{BASE_URL}/v1/chat/completions", json={
    "model": MODEL,
    "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_MSG},
    ],
    "max_tokens": 256,
})
data = resp.json()
print(f"  full response: {data}")
