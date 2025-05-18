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

# Configuración de logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(title="API de Reseñas de Películas", version="1.3")
templates = Jinja2Templates(directory="templates")

df = pd.DataFrame()
sentiment_model = None
analysis_done = False
data_loaded = False
DATA_FILE = "reseñas_procesadas.csv"
RAW_FILE = "reseñas_peliculas_separador_punto_coma.csv"

# Procesar CSV si no existe
def procesar_y_guardar_csv():
    try:
        logger.info("Procesando CSV original y analizando sentimientos...")
        df_raw = pd.read_csv(RAW_FILE, sep=";")

        model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)
        batch_size = 4
        texts = df_raw['razon'].astype(str).tolist()
        sentiments = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                results = model(batch)
                sentiments.extend([
                    "positivo" if res['label'] == "POSITIVE" else "negativo"
                    for res in results
                ])
            except Exception as e:
                logger.warning(f"Error en lote {i}: {str(e)}")
                sentiments.extend(["error"] * len(batch))

        df_raw["sentimiento"] = sentiments
        df_raw.to_csv(DATA_FILE, index=False)
        logger.info(f"Archivo procesado guardado como {DATA_FILE}")
    except Exception as e:
        logger.error(f"Error procesando y guardando CSV: {e}")
        raise RuntimeError("Error en el preprocesamiento")

# Cargar datos
def load_data():
    global df, data_loaded
    try:
        if not os.path.exists(DATA_FILE):
            procesar_y_guardar_csv()
        df = pd.read_csv(DATA_FILE)
        data_loaded = True
        logger.info("Datos cargados desde archivo procesado")
    except Exception as e:
        logger.error(f"Error al cargar datos: {e}")

# Analizar sentimientos solo para validación o reanálisis futuro
def analizar_sentimientos_dummy():
    global analysis_done
    analysis_done = True

@app.get("/health")
def health():
    return {
        "status": "ok",
        "data_loaded": data_loaded,
        "analysis_done": analysis_done
    }

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("inicio.html", {
        "request": request,
        "status": "ready" if analysis_done else "loading",
        "data_ready": data_loaded,
        "total_reseñas": len(df) if data_loaded else 0
    })

@app.get("/formulario", response_class=HTMLResponse)
def mostrar_formulario(request: Request):
    if not data_loaded:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "message": "La base de datos aún no está disponible."
        })

    if not analysis_done:
        return templates.TemplateResponse("espera.html", {
            "request": request,
            "message": "El análisis de sentimientos está en curso...",
            "refresh_interval": 5
        })

    generos = sorted(df['genero'].dropna().unique()) if not df.empty else []
    return templates.TemplateResponse("formulario.html", {
        "request": request,
        "generos": generos,
        "sentimientos": ["positivo", "negativo"]
    })

@app.post("/formulario", response_class=HTMLResponse)
def procesar_formulario(request: Request, sentimiento: Optional[str] = Form(None), genero: Optional[str] = Form(None), top: int = Form(5)):
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

# Inicialización
load_data()
ThreadPoolExecutor().submit(analizar_sentimientos_dummy)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
