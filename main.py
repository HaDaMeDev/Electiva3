# main.py
from fastapi import FastAPI, HTTPException, Query, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
from transformers import pipeline
import torch
from typing import Optional
import os
from concurrent.futures import ThreadPoolExecutor
import logging

# Configuración básica
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API de Reseñas", version="1.1")

# Templates
templates = Jinja2Templates(directory="templates")

# Variables globales
df = None
sentiment_model = None
analysis_done = False

# Carga inicial ligera
@app.on_event("startup")
def startup():
    global df
    try:
        df = pd.read_csv("reseñas_peliculas_separador_punto_coma.csv", sep=";")
        logger.info("Dataset cargado (sin análisis inicial)")
        
        # Inicia análisis en segundo plano
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(analyze_sentiments_background)
    except Exception as e:
        logger.error(f"Error startup: {str(e)}")
        raise

# Análisis en segundo plano
def analyze_sentiments_background():
    global sentiment_model, analysis_done
    
    try:
        logger.info("Iniciando análisis de sentimientos...")
        device = 0 if torch.cuda.is_available() else -1
        
        # Modelo más ligero para producción
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device,
            framework="pt"
        )
        
        # Procesamiento por micro-lotes
        batch_size = 8
        texts = df['razon'].tolist()
        sentiments = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                results = sentiment_model(batch)
                sentiments.extend([
                    "positivo" if res['label'] == "POSITIVE" else "negativo"
                    for res in results
                ])
            except Exception as e:
                logger.warning(f"Error en lote {i}: {str(e)}")
                sentiments.extend(["error"] * len(batch))
                
        df['sentimiento'] = sentiments
        analysis_done = True
        logger.info("Análisis completado!")
        
    except Exception as e:
        logger.error(f"Error análisis: {str(e)}")
        df['sentimiento'] = "error"

# Endpoints
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("inicio.html", {
        "request": request,
        "status": "ready" if analysis_done else "processing"
    })

@app.get("/formulario", response_class=HTMLResponse)
async def show_form(request: Request):
    if not analysis_done:
        return templates.TemplateResponse("loading.html", {
            "request": request,
            "message": "Analizando reseñas... (esto puede tomar 2 minutos)"
        })
    
    generos = sorted(df['genero'].dropna().unique())
    return templates.TemplateResponse("formulario.html", {
        "request": request,
        "generos": generos
    })

# ... (otros endpoints como en tu código original)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)