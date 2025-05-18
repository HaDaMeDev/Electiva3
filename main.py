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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API de Reseñas de Películas", version="1.2")
templates = Jinja2Templates(directory="templates")

df = pd.DataFrame()
sentiment_model = None
analysis_done = False
data_loaded = False

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "data_loaded": data_loaded,
        "analysis_done": analysis_done
    }

def load_data():
    global df, data_loaded
    try:
        df = pd.read_csv("reseñas_peliculas_separador_punto_coma.csv", sep=";")
        data_loaded = True
        logger.info("Datos cargados exitosamente")
        return True
    except Exception as e:
        logger.error(f"Error cargando datos: {str(e)}")
        return False

def analyze_sentiments():
    global sentiment_model, analysis_done
    try:
        logger.info("Iniciando análisis de sentimientos...")
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if torch.cuda.is_available() else -1
        )
        batch_size = 4
        texts = df['razon'].astype(str).tolist()
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
        logger.info("Análisis completado")
    except Exception as e:
        logger.error(f"Error en análisis: {str(e)}")

if load_data():
    ThreadPoolExecutor().submit(analyze_sentiments)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("espera.html" if not analysis_done else "formulario.html", {
        "request": request,
        "message": "Cargando modelo y analizando reseñas..." if not analysis_done else "",
        "generos": sorted(df['genero'].dropna().unique()) if analysis_done else [],
        "sentimientos": ["positivo", "negativo"]
    })

@app.get("/formulario", response_class=HTMLResponse)
async def mostrar_formulario(request: Request):
    if not data_loaded:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "message": "Base de datos no disponible"
        })
    if not analysis_done:
        return templates.TemplateResponse("espera.html", {
            "request": request,
            "message": "El análisis de sentimientos está en progreso...",
        })
    generos = sorted(df['genero'].dropna().unique())
    return templates.TemplateResponse("formulario.html", {
        "request": request,
        "generos": generos,
        "sentimientos": ["positivo", "negativo"]
    })

@app.post("/formulario", response_class=HTMLResponse)
async def procesar_formulario(
    request: Request,
    sentimiento: Optional[str] = Form(None),
    genero: Optional[str] = Form(None),
    top: int = Form(5)
):
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
        logger.error(f"Error procesando formulario: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
