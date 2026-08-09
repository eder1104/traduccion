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
    user_input = (request.text or request.message or "").strip()
    if not user_input:
        return {"response": "Por favor ingresa un mensaje para conversar."}

    # Intentar obtener respuesta conversacional inteligente con la API de IA en la nube
    try:
        system_prompt = f"Responde de manera amigable, conversacional y fluida en el mismo idioma que habla el usuario: {user_input}"
        encoded_prompt = urllib.parse.quote(system_prompt)
        url = f"https://text.pollinations.ai/{encoded_prompt}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            ai_reply = resp.read().decode("utf-8").strip()
            if ai_reply:
                return {"response": ai_reply}
    except Exception:
        pass

    # Fallback conversacional si la API estuviera inaccesible
    try:
        source_lang = detect(user_input)
    except Exception:
        source_lang = "es"

    target_lang = "en" if source_lang == "es" else "es"

    try:
        translated = GoogleTranslator(source='auto', target=target_lang).translate(user_input)
        bot_reply = f"🤖 ¡Hola! Entendí tu mensaje ('{user_input}'). En inglés/español sería: '{translated}'"
    except Exception:
        bot_reply = f"🤖 ¡Hola! Recibí tu mensaje: '{user_input}'. ¿De qué te gustaría hablar?"

    return {"response": bot_reply}
