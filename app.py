from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import random
import requests
import socket
import os
# ⚠️ FIX DNS + timeout
socket.setdefaulttimeout(30)

app = FastAPI()

# ================= HUGGINGFACE API =================
API_URL = "https://api-inference.huggingface.co/models/ThanhPhong123/QuanLyThuVien-ChatBox"
HEADERS = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
}

# ================= LOAD FILE =================
with open("responses.json", "r", encoding="utf-8") as f:
    responses = json.load(f)

with open("label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

# đảo map: id -> intent
label_map = {v: k for k, v in label_map.items()}

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= FALLBACK (khi API chết) =================
def fallback_intent(text):
    text = text.lower()

    if "mở cửa" in text:
        return "gio_mo_cua"
    if "mượn sách" in text:
        return "sach_dang_muon"
    if "lịch sử" in text:
        return "lich_su_muon"

    return "unknown"

# ================= PREDICT =================
def predict(text):
    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={"inputs": text},
            timeout=10
        )

        print("STATUS:", response.status_code)
        print("RAW:", response.text)

        if response.status_code != 200:
            return fallback_intent(text)

        data = response.json()

        if isinstance(data, dict) and "error" in data:
            return fallback_intent(text)

        if not isinstance(data, list) or len(data) == 0:
            return fallback_intent(text)

        best = max(data[0], key=lambda x: x["score"])
        label = best["label"]

        if "LABEL_" in label:
            idx = int(label.split("_")[1])
            return label_map.get(idx, fallback_intent(text))
        else:
            return label

    except Exception as e:
        print("ERROR:", e)
        return fallback_intent(text)  # 👈 QUAN TRỌNG

# ================= RESPONSE =================
def get_response(intent):
    if intent in responses:
        res = responses[intent]

        # nếu là string → trả luôn
        if isinstance(res, str):
            return res

        # nếu là list → random
        if isinstance(res, list):
            return random.choice(res)

    return "Tôi chưa hiểu câu hỏi, bạn có thể nói rõ hơn không?"

# ================= API =================
@app.get("/")
def home():
    return {"status": "OK"}

@app.get("/chat")
def chat(q: str):
    intent = predict(q)
    answer = get_response(intent)

    return {
        "input": q,
        "intent": intent,
        "response": answer
    }