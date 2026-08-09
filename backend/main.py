import os
import re
import urllib.parse
import urllib.request
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from deep_translator import GoogleTranslator
from langdetect import detect

app = FastAPI()

# Configurar carpetas estáticas y plantillas
static_dir = "static" if os.path.exists("static") else "../static"
templates_dir = "templates" if os.path.exists("templates") else static_dir

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates = Jinja2Templates(directory=templates_dir)


class ChatRequest(BaseModel):
    text: str | None = None
    message: str | None = None


def get_natural_conversational_response(text: str) -> str | None:
    t = text.lower().strip()

    # Saludos y preguntas frecuentes en español e inglés
    if re.search(r'\b(como estas|cómo estás|como te va|cómo te va|how are you|how is it going|how do you do)\b', t):
        return "I'm doing great, thank you for asking! How are you doing today?"

    if re.search(r'\b(bien y tu|bien y tú|excelente y tu|genial y tu|good and you|fine and you)\b', t):
        return "I'm doing awesome! What are you up to today?"

    if re.search(r'\b(hola|buenas|saludos|hey|hi|hello)\b', t):
        return "Hello! How are you doing today?"

    if re.search(r'\b(quien eres|quién eres|who are you|tu nombre|what is your name)\b', t):
        return "I'm your AI English conversation partner! I'm here to chat with you and help you practice English."

    if re.search(r'\b(de donde eres|de dónde eres|where are you from)\b', t):
        return "I live in the cloud! Where are you chatting from?"

    if re.search(r'\b(gracias|thank you|thanks|thx)\b', t):
        return "You're very welcome! Feel free to ask or talk about anything else."

    if re.search(r'\b(chao|adios|adiós|bye|goodbye|nos vemos)\b', t):
        return "Goodbye! Have a wonderful day ahead!"

    return None


@app.get("/", response_class=HTMLResponse)
def serve_template(request: Request):
    return templates.TemplateResponse(request=request, name="fase-traductor.html")


@app.get("/chat-page", response_class=HTMLResponse)
def serve_chat_page(request: Request):
    return templates.TemplateResponse(request=request, name="fase-chat.html")


@app.post("/translate")
def translate_text(request: ChatRequest):
    text = (request.text or request.message or "").strip()
    if not text:
        return {"translation": "", "source_lang": "auto", "target_lang": "en"}

    try:
        source_lang = detect(text)
    except Exception:
        source_lang = "auto"

    target_lang = "en" if source_lang == "es" else "es"

    try:
        translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception:
        translated = text

    return {
        "translation": translated,
        "source_lang": source_lang,
        "target_lang": target_lang
    }


@app.post("/chat")
def chat_with_bot(request: ChatRequest):
    user_input = (request.text or request.message or "").strip()
    if not user_input:
        return {"response": "Please enter a message to start our conversation."}

    # 1. Comprobar si es una interacción/pregunta conversacional directa (ej. "cómo estás tú", "hola", "bien y tú")
    natural_match = get_natural_conversational_response(user_input)
    if natural_match:
        return {"response": natural_match}

    # 2. Si no es un saludo básico, consultar a la IA en la nube
    try:
        prompt = f"Reply in 1-2 friendly, natural English sentences to: {user_input}"
        encoded_prompt = urllib.parse.quote(prompt)
        url = f"https://text.pollinations.ai/{encoded_prompt}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        with urllib.request.urlopen(req, timeout=5) as resp:
            ai_reply = resp.read().decode("utf-8").strip()
            if ai_reply and not ai_reply.startswith("An error"):
                return {"response": ai_reply}
    except Exception:
        pass

    # 3. Fallback inteligente y natural en inglés si la red parpadea
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(user_input)
        fallback_reply = f"I see! In English you could say: '{translated}'. What else would you like to discuss?"
    except Exception:
        fallback_reply = f"That's interesting! What else would you like to talk about today?"

    return {"response": fallback_reply}
