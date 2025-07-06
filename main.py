from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import numpy as np

app = FastAPI()

# Load model yang kecil dan ringan
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L3-v2')

model = load_model()

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

class InputQuery(BaseModel):
    tech: str
    user_context: str

class AnswerInput(BaseModel):
    tech: str
    question: str

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI"}

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
    question_embeddings = model.encode(questions, convert_to_numpy=True)
    user_embedding = model.encode(input_data.user_context, convert_to_numpy=True)

    cosine_scores = util.cos_sim(user_embedding, question_embeddings).numpy()
    best_index = int(np.argmax(cosine_scores))

    return {
        "suggested_question": questions[best_index],
        "confidence_score": float(cosine_scores[0][best_index])
    }

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
