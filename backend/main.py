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
        return {"response": "Please enter a message or question."}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    # ── 1. Modelo de IA Autónomo Principal (OpenAI LLM en tiempo real) ──
    try:
        prompt = (
            "You are a smart, autonomous AI assistant. "
            "Always respond in natural, friendly, fluent English to whatever the user says or asks. "
            f"User input: {user_input}\nAI Response:"
        )
        url = f"https://text.pollinations.ai/{urllib.parse.quote(prompt)}?model=openai"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            answer = resp.read().decode("utf-8").strip()
            if answer and not answer.startswith("An error") and len(answer) > 2:
                return {"response": answer}
    except Exception:
        pass

    # ── 2. Modelo de IA Autónomo de Respaldo (Mistral LLM) ──
    try:
        prompt = f"Respond in natural English to: {user_input}"
        url = f"https://text.pollinations.ai/{urllib.parse.quote(prompt)}?model=mistral"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            answer = resp.read().decode("utf-8").strip()
            if answer and not answer.startswith("An error") and len(answer) > 2:
                return {"response": answer}
    except Exception:
        pass

    # ── 3. Fallback en caso de desconexión de red ──
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(user_input)
        return {"response": f"Regarding '{user_input}': {translated}"}
    except Exception:
        return {"response": f"Hello! How can I help you today regarding '{user_input}'?"}
