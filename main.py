from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from transformers import pipeline
import torch
import logging
import uvicorn
from typing import Optional
import os

# Configuración inicial
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API de Reseñas de Películas", version="1.3")
templates = Jinja2Templates(directory="templates")

# Variables de estado (se inicializan una sola vez)
df = pd.DataFrame()
sentiment_model = None
analysis_complete = False
data_loaded = False

# Health Check
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "data_loaded": data_loaded,
        "analysis_complete": analysis_complete
    }

# Carga y análisis único al iniciar
@app.on_event("startup")
def initialize_analysis():
    global df, sentiment_model, analysis_complete, data_loaded
    
    try:
        # 1. Cargar datos
        logger.info("Cargando dataset...")
        df = pd.read_csv("reseñas_peliculas_separador_punto_coma.csv", sep=";")
        data_loaded = True
        
        # 2. Cargar modelo (una sola vez)
        logger.info("Cargando modelo de sentimientos...")
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,  # Forzar CPU
            truncation=True
        )
        
        # 3. Procesamiento completo (una sola ejecución)
        logger.info("Procesando todas las reseñas...")
        texts = df['razon'].astype(str).tolist()
        
        # Procesamiento por lotes optimizado
        batch_size = 8
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
        analysis_complete = True
        logger.info(f"✅ Análisis completado. {len(df)} reseñas procesadas")
        
    except Exception as e:
        logger.error(f"Error durante inicialización: {str(e)}")
        raise

# Endpoints
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("inicio.html", {
        "request": request,
        "status": "ready" if analysis_complete else "loading",
        "total_reseñas": len(df) if data_loaded else 0
    })

@app.get("/formulario", response_class=HTMLResponse)
async def mostrar_formulario(request: Request):
    if not analysis_complete:
        return templates.TemplateResponse("espera.html", {
            "request": request,
            "message": "Análisis inicial en progreso...",
            "refresh_interval": 10  # Auto-refresco cada 10 segundos
        })
    
    generos = sorted(df['genero'].dropna().unique())
    return templates.TemplateResponse("formulario.html", {
        "request": request,
        "generos": generos,
        "sentimientos": ["positivo", "neutral", "negativo"]
    })

@app.post("/formulario", response_class=HTMLResponse)
async def procesar_formulario(
    request: Request,
    sentimiento: Optional[str] = Form(None),
    genero: Optional[str] = Form(None),
    top: int = Form(5)
):
    try:
        # Filtrado sobre datos ya procesados
        df_filtrado = df.copy()
        if sentimiento:
            df_filtrado = df_filtrado[df_filtrado["sentimiento"] == sentimiento]
        if genero:
            df_filtrado = df_filtrado[df_filtrado["genero"].str.lower() == genero.lower()]

        ranking = df_filtrado["pelicula"].value_counts().head(top).to_dict()
        
        return templates.TemplateResponse("resultado.html", {
            "request": request,
            "ranking": ranking,
            "sentimiento": sentimiento,
            "genero": genero,
            "top": top
        })
    except Exception as e:
        logger.error(f"Error procesando formulario: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)