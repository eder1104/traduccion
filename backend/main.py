import os
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

# Historial de conversación para mantener la memoria del chat
conversation_history = []


class ChatRequest(BaseModel):
    text: str | None = None
    message: str | None = None


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
    global conversation_history
    user_input = (request.text or request.message or "").strip()
    if not user_input:
        return {"response": "Please enter a message to start our conversation."}

    # Agregar el mensaje del usuario al historial de conversación
    conversation_history.append(f"User: {user_input}")
    
    # Mantener los últimos 8 turnos de conversación para memoria contextual
    recent_history = "\n".join(conversation_history[-8:])

    # Prompt del sistema para garantizar respuesta autónoma en inglés
    full_prompt = (
        "System Instruction: You are an autonomous, friendly English conversation partner and AI tutor. "
        "Always respond in natural, engaging, fluent English. "
        "React autonomously to what the user says, answer their questions, share ideas, and ask an open-ended follow-up question to keep the conversation going. "
        "If the user speaks in Spanish or another language, respond in English and encourage them to keep practicing English.\n\n"
        f"Conversation History:\n{recent_history}\n\nAI Response in English:"
    )

    try:
        encoded_prompt = urllib.parse.quote(full_prompt)
        url = f"https://text.pollinations.ai/{encoded_prompt}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        
        with urllib.request.urlopen(req, timeout=8) as resp:
            ai_reply = resp.read().decode("utf-8").strip()
            if ai_reply:
                # Guardar respuesta de la IA en la memoria de la conversación
                conversation_history.append(f"AI: {ai_reply}")
                return {"response": ai_reply}
    except Exception:
        pass

    # Fallback conversacional en inglés si la red parpadea
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(user_input)
        fallback_reply = f"That's interesting! Regarding '{translated}', tell me more about your thoughts on this!"
    except Exception:
        fallback_reply = f"I hear you! You said: '{user_input}'. What else would you like to chat about today?"

    conversation_history.append(f"AI: {fallback_reply}")
    return {"response": fallback_reply}
