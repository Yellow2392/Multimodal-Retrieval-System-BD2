import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

DB_HOST = "localhost"
DB_PORT = "5435"
DB_NAME = "motor_multimodal"
DB_USER = "admin"
DB_PASS = "password123"

def get_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        register_vector(conn)
        return conn
    except Exception as e:
        print(f"Error al conectar a la base de datos: {e}")
        return None

def bulk_insert(conn, table_name, columns, values):
    if not values:
        print("No hay datos para insertar.")
        return

    cursor = conn.cursor()
    cols_str = ', '.join(columns)
    
    # molde para la consulta SQL
    template = '(' + ', '.join(['%s'] * len(columns)) + ')'
    query = f"INSERT INTO {table_name} ({cols_str}) VALUES %s"
    
    try:
        execute_values(cursor, query, values, template=template)
        conn.commit()
        print(f"¡Éxito! Se insertaron {len(values)} registros en '{table_name}'.")
    except Exception as e:
        conn.rollback()
        print(f"Error durante el bulk insert en '{table_name}': {e}")
    finally:
        cursor.close()