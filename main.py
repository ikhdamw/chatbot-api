from fastapi import FastAPI
from pydantic import BaseModel
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# === Cohere setup ===
COHERE_API_KEY = "juoSGEbDotqjxOLPLjNwSFfEEDPT82fVPRcv4qLD"
COHERE_URL = "https://api.cohere.ai/v1/embed"
MODEL_NAME = "embed-english-v3.0"

def get_embeddings(texts, input_type="search_document"):
    response = requests.post(
        COHERE_URL,
        headers={
            "Authorization": f"Bearer {COHERE_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL_NAME,
            "texts": texts,
            "input_type": input_type
        }
    )
    res = response.json()
    return res["embeddings"]

# === Static questions & answers ===
tech_questions = {
    "dart": [
        "Apa itu Function dalam Dart?",
        "Bagaimana cara membuat Class di Dart?",
        "Apa perbedaan antara var, final, dan const?",
        "Bagaimana cara menangani Exception di Dart?", 
        "Apa itu Mixin di Dart?",
    ],
    "flutter": [
        "Apa itu Widget dalam Flutter?",
        "Bagaimana cara membuat Stateful Widget?",
        "Apa perbedaan Stateless dan Stateful Widget?",
        "Bagaimana mengatur routing/navigasi?", 
        "Apa itu BuildContext?",
    ]
}

answers = {
    "dart": [
        "Function dalam Dart adalah blok kode yang menjalankan tugas tertentu...",
        "Untuk membuat Class di Dart, gunakan keyword 'class'...",
        "var untuk variabel yang jenisnya bisa berubah...",
        "Exception di Dart ditangani dengan try-catch blocks...",
        "Mixin di Dart adalah cara untuk reuse kode..."
    ],
    "flutter": [
        "Widget di Flutter adalah elemen UI yang membangun aplikasi...",
        "Stateful Widget dibuat dengan class StatefulWidget dan State...",
        "Stateless Widget immutable, Stateful Widget mutable...",
        "Navigasi di Flutter menggunakan Navigator...",
        "BuildContext adalah handle ke lokasi widget..."
    ]
}

# === Models ===
class InputQuery(BaseModel):
    tech: str
    user_context: str

class AnswerInput(BaseModel):
    tech: str
    question: str

# === Routes ===
@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI with Cohere"}

@app.get("/questions")
def get_questions(tech: str):
    tech = tech.lower()
    if tech not in tech_questions:
        return {"error": "Tech tidak dikenali. Gunakan 'dart' atau 'flutter'"}
    return {"questions": tech_questions[tech]}

@app.post("/suggest-question")
def suggest_question(input_data: InputQuery):
    tech = input_data.tech.lower()
    if tech not in tech_questions:
        return {"error": f"Invalid tech value: '{tech}'. Pilih 'dart' atau 'flutter'."}

    questions = tech_questions[tech]

    try:
        # Dapatkan embeddings
        question_embeddings = get_embeddings(questions, input_type="search_document")
        user_embedding = get_embeddings([input_data.user_context], input_type="search_query")[0]

        # Hitung cosine similarity
        scores = cosine_similarity([user_embedding], question_embeddings)[0]

        # Ambil indeks top-N
        max_return = min(3, len(questions))
        top_indices = np.argsort(scores)[::-1][:max_return]

        # Pastikan minimal 1 hasil
        if len(top_indices) < 1:
            return {"error": "Jumlah pertanyaan terlalu sedikit untuk memberi saran."}

        suggestions = []
        for i in top_indices:
            suggestions.append({
                "question": questions[i],
                "confidence_score": float(scores[i])
            })

        return {
            "suggested_questions": suggestions
        }
    except Exception as e:
        return {"error": f"Gagal mendapatkan saran pertanyaan: {str(e)}"}

@app.post("/answer-question")
def answer_question(data: AnswerInput):
    tech = data.tech.lower()
    if tech not in tech_questions:
        return {"error": "Tech tidak dikenali"}

    try:
        index = tech_questions[tech].index(data.question)
    except ValueError:
        return {"error": "Pertanyaan tidak ditemukan"}

    return {"answer": answers[tech][index]}
