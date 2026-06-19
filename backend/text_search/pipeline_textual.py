import re
import pandas as pd
import nltk
import json
import sys
import os

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from collections import Counter

from backend.db_connection import get_connection, bulk_insert

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class TextModules:
    def __init__(self):
        self.stemmer = SnowballStemmer('english')
        self.stop_words = set(stopwords.words('english'))

    # Split
    def split_document(self, text):
        if not isinstance(text, str):
            return []
        
        text_cleaned = re.sub(r'[^a-zA-Z\s]', '', text.lower()).strip()
        return [text_cleaned] if text_cleaned else []

    # Codebook
    def apply_linguistic_techniques(self, clean_chunk): # Tokenizar, eliminar stopwords, stemming
        tokens = nltk.word_tokenize(clean_chunk)
        stems = [self.stemmer.stem(t) for t in tokens if t not in self.stop_words]
        return stems


class SPIMIIndexer:
    def __init__(self, codebook_path, idf_path):# vocabulario y textos idf
        print("Cargando Codebook y pesos IDF en memoria...")
        with open(codebook_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        with open(idf_path, 'r', encoding='utf-8') as f:
            self.idf = json.load(f)
            
        self.modules = TextModules()

    def process_and_index(self, csv_path, chunk_size=500): # Procesa el dataset por bloques y guarda en la bd
        print(f"Iniciando SPIMI Indexer. Leyendo datos de: {csv_path}")
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['description'])
        total_docs = len(df)
        
        conn = get_connection()
        if not conn:
            print("Error: No se pudo conectar a PostgreSQL. Abortando SPIMI.")
            return

        columnas_db = ['producto_id', 'chunk_index', 'contenido_texto', 'term_id', 'tf_idf_score']

        for start_idx in range(0, total_docs, chunk_size):
            end_idx = min(start_idx + chunk_size, total_docs)
            block_df = df.iloc[start_idx:end_idx]
            
            print(f"SPIMI: Procesando bloque de documentos {start_idx} a {end_idx}...")
            spimi_invert_index = []
            
            for _, row in block_df.iterrows():
                doc_id = str(row['id'])
                raw_text = str(row['description'])
                
                # Módulo Split
                chunks = self.modules.split_document(raw_text)
                
                for chunk_idx, chunk in enumerate(chunks):
                    # tokens, stopwords, stemmings
                    stems = self.modules.apply_linguistic_techniques(chunk)
                    
                    if not stems:
                        continue
                        
                    # Contar frecuencias locales para el TF
                    term_counts = Counter(stems)
                    total_terms = len(stems)
                    
                    for term, count in term_counts.items(): # si el término existe en nuestro Codebook global
                        if term in self.vocab:
                            term_id = self.vocab[term]
                            
                            # Moddulo extractor
                            tf = count / total_terms
                            idf_score = self.idf[term]
                            tf_idf_score = tf * idf_score # w = tf * idf
                            
                            spimi_invert_index.append((
                                doc_id, 
                                chunk_idx + 1, 
                                chunk, 
                                term_id, 
                                float(tf_idf_score)
                            ))
                            
            print(f"-> Volcando {len(spimi_invert_index)} registros a la tabla text_chunks...")
            bulk_insert(conn, 'text_chunks', columnas_db, spimi_invert_index) # hacia postgres
            
        conn.close()
        print("¡Indexación SPIMI completada con éxito!")