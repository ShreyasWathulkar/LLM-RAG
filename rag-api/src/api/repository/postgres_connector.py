from dotenv import load_dotenv
import os
import json
import psycopg2
import logging
from typing import Optional, Any

load_dotenv()

class PostgresConnector:
    def __init__(self):
        self.pg_conn_params = json.loads(os.getenv('PG_CONN_PARAMS'))
        self.connection = None
        self.cursor = None

    def connect(self):
        """Establishes a connection to the PostgreSQL database."""
        try:
            self.connection = psycopg2.connect(**self.pg_conn_params)
            self.cursor = self.connection.cursor()
            logging.info("Database connection established successfully.")
        except Exception as e:
            logging.error(f"Error connecting to the database: {e}")
            raise
    
    def disconnect(self):
        """Closes the cursor and connection to the database."""
        if self.cursor:
            self.cursor.close()
            logging.info("Cursor closed.")
        if self.connection:
            self.connection.close()
            logging.info("Database connection closed.")

    def execute_query(self, query: str, params: Optional[tuple] = None, fetch_mode: str = "all") -> Any:
        """Executes a query on the connected database and returns the result based on fetch_mode."""
        if not self.connection:
            raise Exception("Database connection not established. Please connect first.")

        try:
            self.cursor.execute(query, params) if params else self.cursor.execute(query)
            logging.info(f"Executed query: {query}")
            
            # Fetching the result based on fetch_mode
            if fetch_mode == "one":
                return self.cursor.fetchone()  # Fetch a single row
            elif fetch_mode == "all":
                return self.cursor.fetchall()  # Fetch all rows
            else:
                raise ValueError("Invalid fetch_mode. Use 'one' or 'all'.")
        
        except Exception as e:
            logging.error(f"Error executing query: {e}")
            raise

    def execute_non_query(self, query: str, params: Optional[tuple] = None):
        """Executes a non-SELECT query (e.g., INSERT, UPDATE, DELETE)."""
        if not self.connection:
            raise Exception("Database connection not established. Please connect first.")

        try:
            self.cursor.execute(query, params) if params else self.cursor.execute(query)
            self.connection.commit()  # Commit changes for INSERT/UPDATE/DELETE
            logging.info(f"Executed non-query: {query}")
        except Exception as e:
            logging.error(f"Error executing non-query: {e}")
            self.connection.rollback()  # Rollback in case of error
            raise   

    def __enter__(self):
        self.connect()

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()
