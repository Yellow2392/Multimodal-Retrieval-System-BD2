#! Script para crear un sample de 1k imagenes con un csv de descripciones compuestas
# Necesita tener en /dataset/ descargada la data de kaggle (images/, styles/, images.csv, styles.csv)

"""
Recomendaciones para la base de datos:

-- Almacena la metadata de styles.csv. Los dos equipos leemos de aquí
CREATE TABLE IF NOT EXISTS productos (
    id VARCHAR(50) PRIMARY KEY,
    gender VARCHAR(50),
    master_category VARCHAR(100),
    sub_category VARCHAR(100),
    article_type VARCHAR(50),
    base_colour VARCHAR(50),
    season VARCHAR(50),
    year INT,
    usage VARCHAR(50),
    product_display_name TEXT
);

!!! Ojo, styles.csv tiene registros no sanitizados en comas (es un csv delimitado por comas, pero que algunos registros tienen comas dentro sin tener "")
!!! Verificar esos casos al cargar (Pasar ',' a ' ' y los " al caracter de espacio)

-- Para los datos visuales (propuesto)
CREATE TABLE IF NOT EXISTS visual_chunks (
    id SERIAL PRIMARY KEY,
    producto_id VARCHAR(50) REFERENCES productos(id) ON DELETE CASCADE,
    patch_index INT,
    -- SIFT estándar genera vectores matemáticos de 128 dimensiones
    descriptor VECTOR(128),
    -- Columna para guardar la visual word del propio indice invertido
    visual_word_id INT
);

-- Para los datos textuales (propuesto)
CREATE TABLE IF NOT EXISTS text_chunks (
    id SERIAL PRIMARY KEY,
    producto_id VARCHAR(50) REFERENCES productos(id) ON DELETE CASCADE,
    chunk_index INT,
    contenido_texto TEXT,
    -- Columna de tipo TSVECTOR necesaria para probar el índice nativo de PostgreSQL
    vector_nativo TSVECTOR,
    -- Columnas para el propio pipeline unificado (TF-IDF y Codebook)
    term_id INT,
    tf_idf_score FLOAT
);
"""

import os
import json

from backend.text_search.build_codebook import build_global_codebook
from backend.text_search.pipeline_textual import SPIMIIndexer
from backend.text_search.search_engine import TextSearchEngine

DATASET_CSV = "dataset/sample_1k/cleaned_descriptions.csv"
OUTPUT_DIR = "dataset/sample_1k/"
CODEBOOK_JSON = os.path.join(OUTPUT_DIR, "codebook.json")
IDF_JSON = os.path.join(OUTPUT_DIR, "idf_scores.json")

def ejecutar_pipeline_textual():
    print("=== INICIANDO PIPELINE TEXTUAL ===")
    
    if not os.path.exists(CODEBOOK_JSON) or not os.path.exists(IDF_JSON):
        print("Archivos JSON no encontrados. Construyendo Codebook global...")
        build_global_codebook(csv_path=DATASET_CSV, top_k=1000, output_dir=OUTPUT_DIR)
    else:
        print("Codebook ya existente. Saltando fase de entrenamiento lingüístico.")

    print("\n--- Iniciando SPIMI ---")
    indexer = SPIMIIndexer(codebook_path=CODEBOOK_JSON, idf_path=IDF_JSON)
    
    indexer.process_and_index(csv_path=DATASET_CSV, chunk_size=500)
    print("=== PIPELINE TEXTUAL FINALIZADO ===")

if __name__ == "__main__":
    #ejecutar_pipeline_textual()
    # ejecutar_pipeline_visual()
    CODEBOOK_JSON = "dataset/sample_1k/codebook.json"
    IDF_JSON = "dataset/sample_1k/idf_scores.json"
    
    engine = TextSearchEngine(CODEBOOK_JSON, IDF_JSON)
    
    texto_usuario = "Red dress for women"
    print(f"Buscando: '{texto_usuario}'...")
    
    resultados = engine.search(texto_usuario, top_n=3)
    print(json.dumps(resultados, indent=2))