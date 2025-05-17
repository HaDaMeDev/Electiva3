from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from transformers import pipeline
import torch
import logging
from concurrent.futures import ThreadPoolExecutor
import uvicorn

# Configuración inicial
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API de Reseñas de Películas", version="1.1")
templates = Jinja2Templates(directory="templates")

# Variables de estado
df = pd.DataFrame()  # DataFrame vacío por defecto
sentiment_model = None
analysis_done = False
data_loaded = False

# Health Check
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "data_loaded": data_loaded,
        "analysis_done": analysis_done
    }

# Carga de datos mejorada
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

# Análisis de sentimientos optimizado
def analyze_sentiments():
    global sentiment_model, analysis_done
    try:
        logger.info("Iniciando análisis de sentimientos...")
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,
            truncation=True
        )
        
        # Procesamiento por lotes pequeños
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

# Cargar datos al iniciar
if load_data():
    ThreadPoolExecutor().submit(analyze_sentiments)

# Endpoints
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("inicio.html", {
        "request": request,
        "status": "ready" if analysis_done else "loading",
        "data_ready": data_loaded,
        "total_reseñas": len(df) if data_loaded else 0
    })

@app.get("/formulario", response_class=HTMLResponse)
async def show_form(request: Request):
    if not data_loaded:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "message": "Base de datos no disponible"
        })
    
    generos = sorted(df['genero'].dropna().unique()) if not df.empty else []
    return templates.TemplateResponse("formulario.html", {
        "request": request,
        "generos": generos,
        "analysis_done": analysis_done
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)