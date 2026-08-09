import os
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


class TranslationRequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
def serve_template(request: Request):
    return templates.TemplateResponse(request=request, name="fase-traductor.html")


@app.post("/translate")
def translate_text(request: TranslationRequest):
    text = request.text.strip()
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
def chat_with_bot(request: TranslationRequest):
    user_input = request.text.strip()
    if not user_input:
        return {"response": "Por favor ingresa un mensaje válido."}

    try:
        source_lang = detect(user_input)
    except Exception:
        source_lang = "es"

    target_lang = "en" if source_lang == "es" else "es"

    try:
        translated = GoogleTranslator(source='auto', target=target_lang).translate(user_input)
        bot_reply = f"🤖 [Traductor Bot]: {translated}"
    except Exception:
        bot_reply = f"🤖 [Bot]: Recibido: {user_input}"

    return {"response": bot_reply}
