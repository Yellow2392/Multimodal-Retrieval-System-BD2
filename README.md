# Sistema Multimodal de Recuperación y Búsqueda

## Descripción
Este proyecto implementa la base de un sistema unificado de recuperación y búsqueda multimodal sobre texto e imágenes, usando PostgreSQL con pgvector. La arquitectura sigue el enfoque del curso: split de contenido, extracción de características, codebook e índice invertido.

## Objetivo
- Dividir contenido en chunks.
- Extraer descriptores por modalidad.
- Construir representaciones compactas mediante codebook.
- Almacenar y consultar datos en PostgreSQL.
- Comparar el enfoque propio con mecanismos nativos como pgvector.

## Dataset
Se utiliza un dataset de moda con:
- imágenes de productos
- metadatos de `styles.csv`
- descripciones en archivos JSON

El script principal genera una muestra de 1K registros y crea descripciones enriquecidas.

## Estructura del proyecto
- `main.py`: genera la muestra y limpia descripciones.
- `backend/db_connection.py`: conexión a PostgreSQL y carga masiva.
- `backend/pipeline_textual.py`: base para el pipeline textual.
- `backend/pipeline_visual.py`: base para el pipeline visual.
- `docker-compose.yml`: servicio de PostgreSQL con pgvector.

## Requisitos
- Python 3.x
- Docker y Docker Compose
- PostgreSQL con pgvector
- Librerías Python: `pandas`, `psycopg2`, `pgvector`

## Uso
1. Levantar la base de datos con Docker Compose.
2. Ejecutar `main.py` para generar el sample y el CSV limpio.
3. Completar los pipelines textual y visual.
4. Insertar los datos en PostgreSQL.
5. Implementar consultas y comparativas experimentales.

## Estado actual
- Base de datos configurada con Docker.
- Conexión a PostgreSQL implementada.
- Generación de sample y limpieza de datos implementada.
- Pipelines de indexación aún en desarrollo.

## Alcance del proyecto
El sistema está pensado para soportar al menos dos aplicaciones, por ejemplo:
- búsqueda visual de productos
- recomendación multimodal texto-imagen

## Integrantes
- Completar con los nombres del equipo.