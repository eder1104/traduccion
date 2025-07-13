
# Traductor IA + Lenguaje de Señas (MVP)

Este es un MVP de una herramienta de traducción automática con un modo básico para representar el resultado en lenguaje de señas.

## Requisitos

- Python 3.9+
- VS Code con la extensión de Python
- Paquetes: FastAPI, Uvicorn, Transformers, Torch

## Uso

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate    # En Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend
Abre el archivo `frontend/index.html` en tu navegador.

### API Docs
Visita: http://127.0.0.1:8000/docs
