from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer
)
from langdetect import detect
import torch

app = FastAPI()


# Montar carpetas
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Cargar modelo de chat
chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Historial de sesión
chat_sessions = {}

# Modelo de entrada
class TranslationRequest(BaseModel):
    text: str

# Función para obtener el modelo y tokenizer
def get_model_and_tokenizer(source_lang: str, target_lang: str):
    lang_pairs = {
        ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
        ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    }
    key = (source_lang, target_lang)
    model_name = lang_pairs.get(key)
    if not model_name:
        raise ValueError(f"No hay modelo para traducir de {source_lang} a {target_lang}")

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Página principal
@app.get("/", response_class=HTMLResponse)
def serve_template(request: Request):
    return templates.TemplateResponse("fase-traductor.html", {"request": request})

# Ruta de traducción
@app.post("/translate")
def translate_text(request: TranslationRequest):
    text = request.text.strip()

    try:
        source_lang = detect(text)
    except Exception:
        source_lang = "es"

    if source_lang not in ["es", "en"]:
        source_lang = "es"

    target_lang = "en" if source_lang == "es" else "es"

    if source_lang == target_lang:
        return {
            "translation": text,
            "source_lang": source_lang,
            "target_lang": target_lang
        }

    tokenizer, model = get_model_and_tokenizer(source_lang, target_lang)
    inputs = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "translation": translated_text,
        "source_lang": source_lang,
        "target_lang": target_lang
    }

# Ruta del chatbot
@app.post("/chat")
def chat_with_bot(request: TranslationRequest):
    user_input = request.text.strip()
    session_id = "default_user"

    if session_id not in chat_sessions:
        chat_sessions[session_id] = None

    chat_history_ids = chat_sessions[session_id]

    try:
        source_lang = detect(user_input)
    except Exception:
        source_lang = "es"

    if source_lang != "en":
        tokenizer_es_en, model_es_en = get_model_and_tokenizer("es", "en")
        inputs = tokenizer_es_en([user_input], return_tensors="pt", padding=True, truncation=True)
        outputs = model_es_en.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        user_input = tokenizer_es_en.decode(outputs[0], skip_special_tokens=True)

    # Codificar input + historial
    new_input_ids = chat_tokenizer.encode(user_input + chat_tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    chat_history_ids = chat_model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=chat_tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )

    chat_sessions[session_id] = chat_history_ids

    response_en = chat_tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    if source_lang == "es":
        tokenizer_en_es, model_en_es = get_model_and_tokenizer("en", "es")
        inputs = tokenizer_en_es([response_en], return_tensors="pt", padding=True, truncation=True)
        outputs = model_en_es.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        response_es = tokenizer_en_es.decode(outputs[0], skip_special_tokens=True)
        return {"response": response_es}

    return {"response": response_en}
