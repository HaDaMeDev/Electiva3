# main.py
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import logging
from typing import Optional
import os
import uvicorn

# Configuración
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="API de Reseñas de Películas", version="2.0")
templates = Jinja2Templates(directory="templates")

# Variables de estado
df = pd.DataFrame()
data_loaded = False

# Cargar CSV ya procesado
def load_data():
    global df, data_loaded
    try:
        df = pd.read_csv("reseñas_procesadas.csv")
        data_loaded = True
        logger.info("Datos cargados exitosamente.")
        return True
    except Exception as e:
        logger.error(f"Error al cargar datos: {e}")
        return False

# Cargar al iniciar
load_data()

@app.get("/health")
def health_check():
    return {"status": "ok", "data_loaded": data_loaded, "total_reseñas": len(df) if data_loaded else 0}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("inicio.html", {
        "request": request,
        "status": "ready" if data_loaded else "loading",
        "total_reseñas": len(df) if data_loaded else 0
    })

@app.get("/formulario", response_class=HTMLResponse)
async def mostrar_formulario(request: Request):
    if not data_loaded:
        return templates.TemplateResponse("error.html", {"request": request, "message": "Datos no cargados"})
    
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
