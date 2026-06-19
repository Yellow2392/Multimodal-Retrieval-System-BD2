from db_connection import get_connection, bulk_insert

datos_visuales = [
    # (producto_id, patch_index, descriptor_vector), un array de 128 dimensiones
]

conn = get_connection()
if conn:
    columnas = ['producto_id', 'patch_index', 'descriptor']
    bulk_insert(conn, 'visual_chunks', columnas, datos_visuales)
    conn.close()