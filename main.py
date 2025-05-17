from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from transformers import pipeline
import torch
import logging
from typing import Optional
import uvicorn
import os

# Configuración
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API de Reseñas", version="1.3")
templates = Jinja2Templates(directory="templates")

# Variables globales (se ejecutan solo una vez al iniciar)
df = pd.DataFrame()
sentiment_model = None
analysis_complete = False

# Carga y análisis único al iniciar
@app.on_event("startup")
def startup_analysis():
    global df, sentiment_model, analysis_complete
    
    try:
        # 1. Cargar datos
        logger.info("Cargando dataset...")
        df = pd.read_csv("reseñas_peliculas_separador_punto_coma.csv", sep=";")
        
        # 2. Cargar y ejecutar modelo UNA SOLA VEZ
        logger.info("Cargando modelo de sentimientos...")
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",  # Modelo ligero
            device=-1  # Forzar CPU
        )
        
        # 3. Aplicar análisis
        logger.info("Ejecutando análisis...")
        texts = df['razon'].astype(str).tolist()
        df['sentimiento'] = [
            "positivo" if result['label'] == "POSITIVE" else "negativo"
            for result in sentiment_model(texts)
        ]
        
        analysis_complete = True
        logger.info("✅ Análisis completado (una sola ejecución)")
        
    except Exception as e:
        logger.error(f"Error en análisis inicial: {str(e)}")
        raise

# Endpoints
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("inicio.html", {
        "request": request,
        "ready": analysis_complete,
        "total": len(df) if analysis_complete else 0
    })

@app.get("/formulario", response_class=HTMLResponse)
async def show_form(request: Request):
    if not analysis_complete:
        return templates.TemplateResponse("espera.html", {
            "request": request,
            "message": "Análisis inicial en progreso..."
        })
    
    generos = sorted(df['genero'].dropna().unique())
    return templates.TemplateResponse("formulario.html", {
        "request": request,
        "generos": generos
    })

@app.post("/formulario", response_class=HTMLResponse)
async def process_form(
    request: Request,
    genero: str = Form(None),
    sentimiento: str = Form(None)
):
    try:
        df_filtrado = df.copy()
        if genero:
            df_filtrado = df_filtrado[df_filtrado['genero'] == genero]
        if sentimiento:
            df_filtrado = df_filtrado[df_filtrado['sentimiento'] == sentimiento]
            
        resultados = df_filtrado.head(10).to_dict(orient="records")
        
        return templates.TemplateResponse("resultado.html", {
            "request": request,
            "resultados": resultados
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)