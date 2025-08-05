import psycopg2
from psycopg2 import sql, extras
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve PostgreSQL connection parameters from environment variables
PG_CONN_PARAMS = os.getenv('PG_CONN_PARAMS')
pg_conn_params = json.loads(PG_CONN_PARAMS)

def create_chat_history_table():
    """
    Creates the 'chat_history' table in the database with 'session_id' and 'chat_history' columns.
    """
    create_table_query = """
        CREATE TABLE IF NOT EXISTS chat_history (
            session_id VARCHAR(255) PRIMARY KEY,
            chat_history JSONB
        );
    """
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(**pg_conn_params)
        cursor = conn.cursor()
        
        # Execute the create table query
        cursor.execute(create_table_query)
        
        # Commit the changes to the database
        conn.commit()
        
        print("chat_history table created successfully.")
    
    except Exception as e:
        print(f"An error occurred while creating the table: {e}")
    
    finally:
        # Close the cursor and connection
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    create_chat_history_table()
