import os
import re
import json
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


def fetch_wikipedia_summary(topic: str) -> str | None:
    try:
        clean_topic = urllib.parse.quote(topic.strip().replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{clean_topic}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            extract = data.get("extract")
            if extract and len(extract) > 20:
                return extract
    except Exception:
        pass
    return None


def generate_natural_english_chat(user_input: str) -> str:
    t_lower = user_input.lower().strip()

    # 1. Saludos y estado de ánimo cotidianos
    if re.search(r'\b(como estas|cómo estás|como te va|cómo te va|how are you|how do you do)\b', t_lower):
        return "I'm doing fantastic, thank you for asking! How are you doing today?"

    if re.search(r'^\s*(bien|fine|good|excelente|genial)\s*$', t_lower) or re.search(r'\b(bien y tu|bien y tú|good and you|fine and you)\b', t_lower):
        return "I'm glad to hear that! What are your plans for today?"

    if re.search(r'\b(hola|buenas|saludos|hey|hi|hello)\b', t_lower):
        return "Hello! It's great to talk to you. What's on your mind today?"

    if re.search(r'\b(quien eres|quién eres|who are you|tu nombre|what is your name)\b', t_lower):
        return "I'm your AI conversational assistant! I'm here to chat with you and help you practice English."

    if re.search(r'\b(de donde eres|de dónde eres|where are you from)\b', t_lower):
        return "I live in the cloud! Where are you chatting from?"

    if re.search(r'\b(gracias|thank you|thanks)\b', t_lower):
        return "You're very welcome! Let me know if you want to talk about anything else."

    if re.search(r'\b(chao|adios|adiós|bye|goodbye)\b', t_lower):
        return "Goodbye! Have a wonderful day!"

    # 2. Entender la intención traduciéndola al inglés si viene en español
    try:
        translated_en = GoogleTranslator(source='auto', target='en').translate(user_input).strip()
    except Exception:
        translated_en = user_input

    en_lower = translated_en.lower()

    # 3. Preguntas sobre gustos ("te gusta X?", "do you like X?")
    like_match = re.search(r'\b(do you like|do u like|like|te gusta)\s+(.+)', en_lower)
    if like_match:
        item = like_match.group(2).strip('? .!')
        if "coffee" in item or "cafe" in item or "café" in item:
            return "I love coffee! As an AI I don't drink it, but it's one of the most popular drinks in the world. Do you prefer espresso, cappuccino, or black coffee?"
        return f"I find {item} really fascinating! As an AI I don't experience it the human way, but tell me, what do you love about {item}?"

    # 4. Preguntas sobre favoritos ("cuál es tu X favorito?", "what is your favorite X?")
    fav_match = re.search(r'\b(what is your favorite|what\'s your favorite|favorite|favorito)\s+(.+)', en_lower)
    if fav_match:
        thing = fav_match.group(2).strip('? .!')
        return f"I find many types of {thing} interesting! What about you, what's your top favorite {thing}?"

    # 5. Preguntas sobre actividad o capacidades ("qué haces?", "puedes X?")
    if re.search(r'\b(what are you doing|que haces|qué haces|what do you do)\b', en_lower):
        return "I'm right here chatting with you and ready to talk about anything! What are you up to today?"

    if re.search(r'\b(can you|puedes)\b', en_lower):
        return "Yes! I can help you discuss topics, practice English, translate text, and answer your questions. What would you like to try?"

    # 6. Consultas sobre conocimiento enciclopédico ("qué es X", "quién es Y")
    search_topic = re.sub(r'^(what is|who is|explain|tell me about|que es|quien es|explicame|cuentame de)\s+', '', translated_en, flags=re.IGNORECASE).strip('? .!')
    if len(search_topic) > 2 and search_topic.lower() != translated_en.lower():
        wiki_res = fetch_wikipedia_summary(search_topic)
        if wiki_res:
            return f"Here is what I know about {search_topic.capitalize()}:\n{wiki_res}"

    # 7. Respuesta conversacional natural abierta
    return f"That's interesting! Speaking about '{translated_en}', what else would you like to discuss today?"


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

    response_text = generate_natural_english_chat(user_input)
    return {"response": response_text}
