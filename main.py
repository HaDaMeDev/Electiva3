from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from transformers import pipeline
import torch
import logging
import uvicorn
import os
from typing import Optional

# Configuración
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API de Reseñas", version="1.4")
templates = Jinja2Templates(directory="templates")

# Variables de estado
app.state.df = pd.DataFrame()
app.state.model = None
app.state.ready = False

# Health Check mejorado
@app.get("/health", response_class=JSONResponse)
async def health_check():
    if app.state.ready:
        return {"status": "ready"}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "message": "Aplicación iniciando..."}
        )

# Inicialización optimizada
@app.on_event("startup")
async def startup_event():
    try:
        # 1. Cargar datos
        logger.info("Cargando dataset...")
        app.state.df = pd.read_csv("reseñas_peliculas_separador_punto_coma.csv", sep=";")
        
        # 2. Cargar modelo (versión ligera)
        logger.info("Cargando modelo...")
        app.state.model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,
            framework="pt",
            truncation=True
        )
        
        # 3. Procesamiento inicial
        logger.info("Procesando reseñas...")
        texts = app.state.df['razon'].astype(str).tolist()
        app.state.df['sentimiento'] = [
            "positivo" if result['label'] == "POSITIVE" else "negativo"
            for result in app.state.model(texts[:1000])  # Procesa solo las primeras 1000 para demo
        ]
        
        app.state.ready = True
        logger.info("✅ Servicio listo")
        
    except Exception as e:
        logger.error(f"Error de inicialización: {str(e)}")
        raise

# Endpoint raíz
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("inicio.html", {
        "request": request,
        "ready": app.state.ready,
        "total": len(app.state.df)
    })

# Resto de endpoints (formulario, etc.)...

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        timeout_keep_alive=600  # Aumentado para inicialización
    )