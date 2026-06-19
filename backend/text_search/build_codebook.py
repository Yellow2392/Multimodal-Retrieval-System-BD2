import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import math
import json
import os
from collections import defaultdict

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def build_global_codebook(csv_path, top_k=5000, output_dir="../dataset/sample_1k/"):# vocabulario y texto idf globales
    print("Iniciando construcción del Codebook global...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['description'])
    total_docs = len(df)
    
    stemmer = SnowballStemmer('english') # todas las descripciones están en inglés
    stop_words = set(stopwords.words('english'))
    
    # en cuántos documentos aparece cada palabra
    document_frequency = defaultdict(int)
    
    print(f"Procesando {total_docs} documentos para extraer frecuencias...")
    
    for text in df['description']:
        # tokenizar, eliminar stopwords, stemming
        tokens = nltk.word_tokenize(str(text).lower())
        
        # stems únicos por documento -> DF (Document Frequency)
        unique_stems_in_doc = set([
            stemmer.stem(t) for t in tokens 
            if t.isalpha() and t not in stop_words
        ])
        
        for stem in unique_stems_in_doc:
            document_frequency[stem] += 1
            
    # Top K
    sorted_terms = sorted(document_frequency.items(), key=lambda x: x[1], reverse=True) # por frecuencia de documentos
    top_terms = sorted_terms[:top_k]
    
    #IDF Global
    # IDF = log(N / DF)
    codebook = {}
    idf_scores = {}
    
    for term_id, (term, df_count) in enumerate(top_terms):
        codebook[term] = term_id
        
        # Frecuencia Inversa de documento
        idf = math.log10(total_docs / float(df_count))
        idf_scores[term] = idf
        
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "codebook.json"), "w") as f:
        json.dump(codebook, f, indent=4)
        
    with open(os.path.join(output_dir, "idf_scores.json"), "w") as f:
        json.dump(idf_scores, f, indent=4)
        
    print(f"¡Codebook generado con éxito!")
    print(f"Total de términos en el vocabulario: {len(codebook)}")
    print(f"Archivos guardados en: {output_dir}")