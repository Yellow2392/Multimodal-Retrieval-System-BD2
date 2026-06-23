from db_connection import get_connection, bulk_insert

datos_texto = [
    # (producto_id, chunk_index, contenido_texto, term_id, tf_idf_score)
]

conn = get_connection()
if conn:
    columnas = ['producto_id', 'chunk_index', 'contenido_texto', 'term_id', 'tf_idf_score']
    bulk_insert(conn, 'text_chunks', columnas, datos_texto)
    conn.close()
    