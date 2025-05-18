# procesar_csv.py
import pandas as pd
from transformers import pipeline
import torch

# Cargar archivo original
df = pd.read_csv("reseñas_peliculas_separador_punto_coma.csv", sep=";")

# Preparar modelo de sentimiento
device = 0 if torch.cuda.is_available() else -1
modelo = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)

# Analizar sentimientos por lotes
batch_size = 4
textos = df["razon"].astype(str).tolist()
sentimientos = []

for i in range(0, len(textos), batch_size):
    batch = textos[i:i+batch_size]
    try:
        resultados = modelo(batch)
        sentimientos.extend(["positivo" if r["label"] == "POSITIVE" else "negativo" for r in resultados])
    except Exception as e:
        print(f"Error en lote {i}: {e}")
        sentimientos.extend(["error"] * len(batch))

# Guardar resultados
df["sentimiento"] = sentimientos
df.to_csv("reseñas_procesadas.csv", index=False)
print("Archivo 'reseñas_procesadas.csv' generado correctamente.")
