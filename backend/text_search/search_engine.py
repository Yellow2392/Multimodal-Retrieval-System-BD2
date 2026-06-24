import json
import psycopg2
from collections import Counter
import os
import sys
import re

from .pipeline_textual import TextModules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from db_connection import get_connection

class TextSearchEngine:
    def __init__(self, codebook_path, idf_path): # Carga el vocabulario y pesos para procesar las búsquedas en tiempo real
        with open(codebook_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        with open(idf_path, 'r', encoding='utf-8') as f:
            self.idf = json.load(f)
            
        self.modules = TextModules()

    def search(self, query_string, top_n=5, mode='manual'): # Si es modo nativo usa GIN + tsvector
        if mode == 'native':
            return self._search_native(query_string, top_n)
        else:
            return self._search_manual(query_string, top_n)

    def _search_native(self, query_string, top_n):
        # GIN + tsvector
        clean_text = re.sub(r'[^a-zA-Z\s]', '', query_string.lower())
        tokens = clean_text.split()
        if not tokens:
            return []
            
        ts_query_string = ' | '.join(tokens)
        
        conn = get_connection()
        if not conn:
            return [{"error": "No hay conexión a la base de datos"}]

        cursor = conn.cursor()
        
        sql_query = """
            SELECT 
                p.id, 
                p.product_display_name, 
                p.base_colour,
                p.master_category,
                MAX(ts_rank(tc.vector_nativo, to_tsquery('english', %s))) AS similarity_score
            FROM text_chunks tc
            JOIN productos p ON p.id = tc.producto_id
            WHERE tc.vector_nativo @@ to_tsquery('english', %s)
            GROUP BY p.id, p.product_display_name, p.base_colour, p.master_category
            ORDER BY similarity_score DESC
            LIMIT %s;
        """
        
        try:
            cursor.execute(sql_query, (ts_query_string, ts_query_string, top_n))
            resultados = cursor.fetchall()
            
            response = []
            for row in resultados:
                response.append({
                    "id": row[0],
                    "name": row[1],
                    "color": row[2],
                    "category": row[3],
                    "score": round(row[4], 4),
                    "image_url": f"/dataset/images/{row[0]}.jpg"
                })
            return response
            
        except Exception as e:
            print(f"Error en la búsqueda nativa: {e}")
            return []
        finally:
            cursor.close()
            conn.close()

    def _search_manual(self, query_string, top_n):
        # SPIMI + TF-IDF normalizado
        chunks = self.modules.split_document(query_string)
        if not chunks:
            return []
            
        stems = self.modules.apply_linguistic_techniques(chunks[0])
        if not stems:
            return []

        term_counts = Counter(stems)
        total_terms = len(stems)
        
        query_term_ids = [] # calcular tf-idf
        query_tf_idf_scores = []
        
        for term, count in term_counts.items():
            if term in self.vocab:
                term_id = self.vocab[term]
                tf = count / total_terms
                idf_score = self.idf[term]
                tf_idf_score = tf * idf_score
                
                query_term_ids.append(term_id)
                query_tf_idf_scores.append(float(tf_idf_score))

        if not query_term_ids: # si se busca palabras que no estén en vocabulario
            return []

        conn = get_connection()
        if not conn:
            return [{"error": "No hay conexión a la base de datos"}]

        cursor = conn.cursor()
        
        # Usa las listas generadas para multiplicar los pesos 
        # (tf-idf busqueda x tf-idf documento) y ordenarlos de mayor a menor relevancia
        sql_query = """
            WITH query_terms AS (
                SELECT unnest(%s::int[]) AS term_id, 
                    unnest(%s::float[]) AS q_score
            ),
            document_norms AS (
                -- Calcula la norma euclidiana (longitud del vector) para cada producto
                SELECT 
                    producto_id, 
                    SQRT(SUM(tf_idf_score * tf_idf_score)) AS doc_norm
                FROM text_chunks
                GROUP BY producto_id
            )
            SELECT 
                p.id, 
                p.product_display_name, 
                p.base_colour,
                p.master_category,
                -- Normalizacion coseno
                SUM(tc.tf_idf_score * qt.q_score) / dn.doc_norm AS similarity_score
            FROM text_chunks tc
            JOIN query_terms qt ON tc.term_id = qt.term_id
            JOIN productos p ON p.id = tc.producto_id
            JOIN document_norms dn ON dn.producto_id = tc.producto_id
            GROUP BY p.id, p.product_display_name, p.base_colour, p.master_category, dn.doc_norm
            ORDER BY similarity_score DESC
            LIMIT %s;
        """
        
        try:
            cursor.execute(sql_query, (query_term_ids, query_tf_idf_scores, top_n))
            resultados = cursor.fetchall()
            
            response = []
            for row in resultados:
                response.append({
                    "id": row[0],
                    "name": row[1],
                    "color": row[2],
                    "category": row[3],
                    "score": round(row[4], 4),
                    "image_url": f"/dataset/images/{row[0]}.jpg" 
                })
            return response
            
        except Exception as e:
            print(f"Error en la búsqueda manual: {e}")
            return []
        finally:
            cursor.close()
            conn.close()