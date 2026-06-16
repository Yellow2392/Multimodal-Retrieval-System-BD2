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
!!! Verificar esos casos al cargar (Pasar ',' a ' ')

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
import shutil
import json
import re
import pandas as pd
import random

def clean_html(raw_html):
    if not isinstance(raw_html, str):
        return ""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return ' '.join(cleantext.split())

def create_sample_dataset(source_dir="dataset", sample_size=1000):
    images_dir = os.path.join(source_dir, "images")
    styles_dir = os.path.join(source_dir, "styles")
    styles_csv_path = os.path.join(source_dir, "styles.csv")
    
    sample_dir = os.path.join(source_dir, "sample_1k")
    sample_images_dir = os.path.join(sample_dir, "images")
    os.makedirs(sample_images_dir, exist_ok=True)
    
    print("Cargando metadatos de styles.csv...")
    df_styles = pd.read_csv(styles_csv_path, on_bad_lines='skip')
    df_styles['id'] = df_styles['id'].astype(str) 
    styles_dict = df_styles.set_index('id').to_dict('index')
    
    all_json_files = [f for f in os.listdir(styles_dir) if f.endswith('.json')]
    available_ids = [os.path.splitext(f)[0] for f in all_json_files]
    valid_ids = [img_id for img_id in available_ids if os.path.exists(os.path.join(images_dir, f"{img_id}.jpg"))]
    
    sampled_ids = random.sample(valid_ids, sample_size) if len(valid_ids) >= sample_size else valid_ids
    print(f"Procesando muestra de {len(sampled_ids)} registros...")
    
    cleaned_data = []
    
    for img_id in sampled_ids:
        src_image = os.path.join(images_dir, f"{img_id}.jpg")
        dst_image = os.path.join(sample_images_dir, f"{img_id}.jpg")
        shutil.copy2(src_image, dst_image)
        
        csv_metadata = ""
        if img_id in styles_dict:
            row = styles_dict[img_id]
            fields = [str(row.get(col, '')) for col in ['gender', 'masterCategory', 'subCategory', 'baseColour', 'productDisplayName']]
            csv_metadata = " ".join([f for f in fields if f and f != 'nan'])

        json_path = os.path.join(styles_dir, f"{img_id}.json")
        clean_text = ""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "data" in data and "productDescriptors" in data["data"]:
                    descriptors = data["data"]["productDescriptors"]
                    if "description" in descriptors and "value" in descriptors["description"]:
                        clean_text = clean_html(descriptors["description"]["value"])
        except Exception as e:
            pass 
            
        final_text = f"{csv_metadata} {clean_text}".strip()
        
        cleaned_data.append({
            "id": img_id,
            "description": final_text
        })

    csv_path = os.path.join(sample_dir, "cleaned_descriptions.csv")
    pd.DataFrame(cleaned_data).to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Descripciones enriquecidas guardadas en: {csv_path}")

if __name__ == "__main__":
    create_sample_dataset()