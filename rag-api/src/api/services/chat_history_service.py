import logging
import json
from typing import List, Dict
from repository.postgres_connector import PostgresConnector




class ChatHistoryService:

    def __init__(self, db_connector: PostgresConnector):
        self.db_connector = db_connector        

    def retrieve_chat_history(self, session_id):
        try:
            # with self.db_connector:
            result = self.db_connector.execute_query(
                "SELECT chat_history FROM chat_history WHERE session_id = %s", 
                (session_id,),
                fetch_mode="one"  # Use fetchone for a single row
            )

            if result:
                # If chat history exists, parse it and set to chat_history
                chat_history = result[0]
                return chat_history
            else:
                # If no chat history found, initialize as an empty list
                chat_history = []
                return chat_history
                    
        except Exception as e:
            logging.error(f"Error retrieving chat history for session_id {session_id}: {e}")
            # If an error occurs, initialize an empty chat history
            chat_history = []
            return chat_history

            

    def store_chat_history(self, chat_history: List[Dict[str, str]], session_id: str):
        chat_history_json = json.dumps(chat_history)

        try:
            # with self.db_connector:
                # Execute the insert/update query
            self.db_connector.execute_non_query(
                """
                INSERT INTO chat_history (session_id, chat_history)
                VALUES (%s, %s)
                ON CONFLICT (session_id) 
                DO UPDATE SET chat_history = EXCLUDED.chat_history;
                """,
                (session_id, chat_history_json)
            )

            logging.info(f"Chat history for session {session_id} saved to the database.")
        
        except Exception as e:
            logging.error(f"Failed to save chat history for session {session_id}: {e}")

    @staticmethod
    def update_chat_history(chat_history, user_question, ai_response) -> List[Dict[str, str]]:
        chat_history.append({"role": "human", "content": user_question})
        chat_history.append({"role": "assistant", "content": ai_response})

        return chat_history