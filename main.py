from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from transformers import pipeline
import torch
import logging
from concurrent.futures import ThreadPoolExecutor
import uvicorn
from typing import Optional
import os

# Configuración inicial
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API de Reseñas de Películas", version="1.2")
templates = Jinja2Templates(directory="templates")

# Variables globales
df = pd.DataFrame()
sentiment_model = None
data_loaded = False
analysis_done = False

# Función para cargar el CSV
def load_data():
    global df, data_loaded
    try:
        df = pd.read_csv("reseñas_peliculas_separador_punto_coma.csv", sep=";")
        data_loaded = True
        logger.info("Datos cargados correctamente.")
    except Exception as e:
        logger.error(f"Error cargando CSV: {e}")

# Función para procesar análisis de sentimientos
def analyze_sentiments():
    global sentiment_model, analysis_done, df
    try:
        logger.info("Iniciando análisis de sentimientos...")
        sentiment_model = pipeline("sentiment-analysis",
                                   model="distilbert-base-uncased-finetuned-sst-2-english",
                                   device=0 if torch.cuda.is_available() else -1)
        sentiments = []
        for razon in df['razon'].astype(str):
            try:
                resultado = sentiment_model(razon[:512])[0]
                sentimiento = "positivo" if resultado['label'] == "POSITIVE" else "negativo"
            except:
                sentimiento = "error"
            sentiments.append(sentimiento)
        df['sentimiento'] = sentiments
        analysis_done = True
        logger.info("Análisis completado.")
    except Exception as e:
        logger.error(f"Error en análisis de sentimientos: {e}")

# Cargar datos y lanzar análisis una sola vez
if load_data() is None:
    ThreadPoolExecutor().submit(analyze_sentiments)

# Health check
@app.get("/health")
def health():
    return {
        "status": "ok",
        "data_loaded": data_loaded,
        "analysis_done": analysis_done,
        "total_reseñas": len(df)
    }

@app.get("/", response_class=HTMLResponse)
async def inicio(request: Request):
    return templates.TemplateResponse("inicio.html", {
        "request": request,
        "status": "ready" if analysis_done else "loading",
        "data_ready": data_loaded,
        "total_reseñas": len(df) if data_loaded else 0
    })

@app.get("/formulario", response_class=HTMLResponse)
async def formulario(request: Request):
    if not data_loaded:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "message": "Datos no disponibles"
        })
    if not analysis_done:
        return templates.TemplateResponse("espera.html", {
            "request": request,
            "message": "Análisis en progreso...",
            "refresh_interval": 5
        })
    generos = sorted(df["genero"].dropna().unique())
    return templates.TemplateResponse("formulario.html", {
        "request": request,
        "generos": generos,
        "sentimientos": ["positivo", "negativo"]
    })

@app.post("/formulario", response_class=HTMLResponse)
async def procesar(request: Request,
                   sentimiento: Optional[str] = Form(None),
                   genero: Optional[str] = Form(None),
                   top: int = Form(5)):
    try:
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
        logger.error(f"Error al procesar el formulario: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
