from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import random
import requests
import socket
import os
import time
socket.setdefaulttimeout(30)
app = FastAPI()
API_URL = "https://api-inference.huggingface.co/models/ThanhPhong123/QuanLyThuVien-ChatBox"

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("WARNING: HF_TOKEN is missing!")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}
with open("responses.json", "r", encoding="utf-8") as f:
    responses = json.load(f)

with open("label_map.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

label_map = {v: k for k, v in label_map.items()}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
def fallback_intent(text):
    text = text.lower()

    if "mở cửa" in text or "đóng cửa" in text:
        return "gio_mo_cua"

    if "mượn sách" in text:
        return "sach_dang_muon"

    if "lịch sử" in text:
        return "lich_su_muon"

    return "unknown" 

def predict(text):
    try:
        for attempt in range(3):
            print(f"\nCall API (attempt {attempt+1})")

            response = requests.post(
                API_URL,
                headers=HEADERS,
                json={"inputs": text},
                timeout=20
            )

            print("STATUS:", response.status_code)
            print("RAW:", response.text)
            if response.status_code != 200:
                print("HTTP ERROR → fallback")
                return fallback_intent(text)

            data = response.json()
            if isinstance(data, dict) and "error" in data:
                error_msg = data["error"].lower()

                if "loading" in error_msg:
                    print("⏳ Model loading... đợi 5s")
                    time.sleep(5)
                    continue 

                print("API ERROR:", data["error"])
                return fallback_intent(text)
            if not isinstance(data, list) or len(data) == 0:
                print("INVALID DATA → fallback")
                return fallback_intent(text)
            best = max(data[0], key=lambda x: x["score"])
            label = best["label"]

            print("PREDICT:", label)

            if "LABEL_" in label:
                idx = int(label.split("_")[1])
                return label_map.get(idx, fallback_intent(text))

            return label
        print("FAIL SAU 3 LẦN → fallback")
        return fallback_intent(text)

    except Exception as e:
        print("EXCEPTION:", e)
        return fallback_intent(text)
def get_response(intent):
    if intent in responses:
        res = responses[intent]

        if isinstance(res, str):
            return res

        if isinstance(res, list):
            return random.choice(res)

    return "Tôi chưa hiểu câu hỏi, bạn có thể nói rõ hơn không?"
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