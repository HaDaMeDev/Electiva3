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
import uvicorn

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API de Reseñas de Películas", version="1.0")

# Configuración inicial
templates = Jinja2Templates(directory="templates")
df = None
sentiment_model = None
analysis_done = False
executor = ThreadPoolExecutor(max_workers=1)

# Health Check para Railway
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "analysis_ready": analysis_done,
        "model_loaded": sentiment_model is not None
    }

# Cargar datos sin análisis inicial
def load_data():
    global df
    try:
        df = pd.read_csv("reseñas_peliculas_separador_punto_coma.csv", sep=";")
        logger.info("Dataset cargado (sin análisis inicial)")
    except Exception as e:
        logger.error(f"No se pudo cargar el CSV: {str(e)}")
        raise RuntimeError(f"No se pudo cargar el CSV: {str(e)}")

# Análisis de sentimientos optimizado para Railway
def analyze_sentiments():
    global df, sentiment_model, analysis_done
    try:
        logger.info("Iniciando análisis de sentimientos en segundo plano...")
        device = -1  # Forzamos CPU para evitar problemas en Railway
        
        # Usamos un modelo más ligero para producción
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device,
            truncation=True
        )
        
        # Procesamiento por lotes pequeños
        batch_size = 4
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
        logger.info("Análisis de sentimientos completado")
    except Exception as e:
        logger.error(f"Error en análisis: {str(e)}")
        df['sentimiento'] = "error"

# Cargar datos al iniciar
load_data()

# Endpoints originales
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("inicio.html", {
        "request": request,
        "status": "ready" if analysis_done else "loading"
    })

@app.get("/formulario", response_class=HTMLResponse)
async def mostrar_formulario(request: Request):
    # Iniciar análisis en segundo plano si es la primera consulta
    if not analysis_done and not sentiment_model:
        executor.submit(analyze_sentiments)
    
    generos = sorted(df['genero'].dropna().unique())
    return templates.TemplateResponse(
        "formulario.html",
        {
            "request": request,
            "generos": generos,
            "sentimientos": ["positivo", "neutral", "negativo"],
            "analysis_done": analysis_done
        }
    )

@app.post("/formulario", response_class=HTMLResponse)
async def procesar_formulario(
    request: Request,
    sentimiento: Optional[str] = Form(None),
    genero: Optional[str] = Form(None),
    top: int = Form(5)
):
    if not analysis_done:
        return templates.TemplateResponse(
            "espera.html",
            {
                "request": request,
                "mensaje": "El análisis de sentimientos está en progreso...",
                "refresh_interval": 5  # Segundos para auto-recarga
            }
        )
    
    try:
        df_filtrado = df.copy()
        if sentimiento:
            df_filtrado = df_filtrado[df_filtrado["sentimiento"] == sentimiento]
        if genero:
            df_filtrado = df_filtrado[df_filtrado["genero"].str.lower() == genero.lower()]

        ranking = df_filtrado["pelicula"].value_counts().head(top).to_dict()
        
        return templates.TemplateResponse(
            "resultado.html",
            {
                "request": request,
                "ranking": ranking,
                "sentimiento": sentimiento,
                "genero": genero,
                "top": top
            }
        )
    except Exception as e:
        logger.error(f"Error procesando formulario: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno")

# Endpoint para API JSON
@app.get("/reseñas")
def obtener_reseñas(skip: int = 0, limit: int = 10):
    if not analysis_done:
        raise HTTPException(status_code=503, detail="Análisis en progreso")
    return df.iloc[skip:skip + limit].to_dict(orient="records")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        timeout_keep_alive=300  # Mayor tiempo para análisis
    )