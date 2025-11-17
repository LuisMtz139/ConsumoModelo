from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("ERROR: No se encontró OPENAI_API_KEY en el archivo .env")

client = OpenAI(api_key=API_KEY)

app = FastAPI(
    title="Prediction AI (Modelo Económico)",
    version="1.0"
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):

    respuesta = client.chat.completions.create(
        model="gpt-4.1-mini",  # Modelo económico
        temperature=0.4,       # Bajo consumo + respuestas estables
        max_tokens=300,        # Limita la respuesta (ahorra tokens)
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un asistente muy eficiente y económico. "
                    "Da respuestas claras y cortas para ahorrar tokens. "
                    "Ayudas con ERP Prediction, dudas técnicas y temas generales."
                )
            },
            {"role": "user", "content": req.message}
        ]
    )

    reply_text = respuesta.choices[0].message.content
    return ChatResponse(reply=reply_text)
