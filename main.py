from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import torch

app = FastAPI()

# Load model sekali saja
model = SentenceTransformer('all-MiniLM-L6-v2')

# Pertanyaan berdasarkan HTML
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


class InputQuery(BaseModel):
    tech: str  # "dart" atau "flutter"
    user_context: str  # teks bebas dari user

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI on Railway"}

@app.post("/suggest-question")
def suggest_question(input_data: InputQuery):
    tech = input_data.tech.lower()
    if tech not in tech_questions:
        return {
            "error":
            f"Invalid tech value: '{tech}'. Pilih 'dart' atau 'flutter'."
        }

    questions = tech_questions[tech]
    question_embeddings = model.encode(questions, convert_to_tensor=True)
    user_embedding = model.encode(input_data.user_context,
                                  convert_to_tensor=True)

    cosine_scores = util.cos_sim(user_embedding, question_embeddings)
    best_index = torch.argmax(cosine_scores).item()

    return {
        "suggested_question": questions[best_index],
        "confidence_score": float(cosine_scores[0][best_index])
    }
