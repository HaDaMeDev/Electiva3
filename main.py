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

app = FastAPI(title="API de Reseñas de Películas", version="1.0")

# Configuración inicial
templates = Jinja2Templates(directory="templates")
df = None
sentiment_model = None
analysis_done = False
executor = ThreadPoolExecutor(max_workers=1)

# Cargar datos sin análisis inicial
def load_data():
    global df
    try:
        df = pd.read_csv("reseñas_peliculas_separador_punto_coma.csv", sep=";")
        print("Dataset cargado (sin análisis inicial)")
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar el CSV: {str(e)}")

# Análisis de sentimientos (se ejecutará en segundo plano)
def analyze_sentiments():
    global df, sentiment_model, analysis_done
    try:
        print("Iniciando análisis de sentimientos en segundo plano...")
        device = 0 if torch.cuda.is_available() else -1
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=device
        )
        
        sentimientos = []
        for texto in df['razon']:
            try:
                resultado = sentiment_model(texto[:512])[0]
                rating = int(resultado['label'][0])
                sentimiento = "positivo" if rating >= 4 else "neutral" if rating == 3 else "negativo"
            except:
                sentimiento = "error"
            sentimientos.append(sentimiento)
        
        df['sentimiento'] = sentimientos
        analysis_done = True
        print("Análisis de sentimientos completado")
    except Exception as e:
        print(f"Error en análisis: {str(e)}")
        df['sentimiento'] = "error"

# Cargar datos inmediatamente al iniciar
load_data()

# Endpoints
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("inicio.html", {"request": request})

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
            "sentimientos": ["positivo", "neutral", "negativo"]
        }
    )

@app.post("/formulario", response_class=HTMLResponse)
async def procesar_formulario(
    request: Request,
    sentimiento: Optional[str] = Form(None),
    genero: Optional[str] = Form(None),
    top: int = Form(5)
):
    # Mostrar resultados aunque el análisis no haya terminado
    if not analysis_done:
        return templates.TemplateResponse(
            "espera.html",
            {"request": request, "mensaje": "El análisis de sentimientos está en progreso..."}
        )
    
    # Procesar filtros
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

# Resto de endpoints (reseñas, analizar, estadisticas, etc.)...