from dotenv import load_dotenv
import logging
import json
from langchain.schema import StrOutputParser
from operator import itemgetter
from models.rag_config import *
from models.qa_model import QaInputModel
from services.chat_history_service import ChatHistoryService
from repository.postgres_connector import PostgresConnector
from langfuse.callback import CallbackHandler

load_dotenv()

langfuse_handler = CallbackHandler(user_id="madhav")


class RAGService():
    def __init__(self, llm, history_aware_retriever_prompt, RAG_prompt, embedding_model, db_connector, chat_history_service):
        self.llm = llm
        self.history_aware_retriever_prompt = history_aware_retriever_prompt
        self.RAG_prompt = RAG_prompt
        self.embedding_model = embedding_model
        self.chat_history = []
        self.history_aware_retriever = self.create_history_aware_retriever() 
        self.chain = self.create_chain()
        self.db_connector = db_connector
        self.chat_history_service = chat_history_service

    def get_embedding(self, user_query):

        response = self.embedding_model

        embedding = response.encode(user_query)

        # Converting the embedding to the pgvector and returning it
        return '[' + ','.join(map(str, embedding)) + ']'

    def context_retriever(self, user_query):
        # with self.db_connector:
        # Get prompt vector for the user query
        prompt_vector = self.get_embedding(user_query)
        
        # Execute a non-query for setting max_parallel_workers_per_gather
        self.db_connector.execute_non_query("SET max_parallel_workers_per_gather = 4")
        
        # Execute the main query with the prompt vector and retrieve the result
        query = """
            SELECT text, url 
            FROM madhav_news_scroll11 
            WHERE 1 - (embeddings_mxdbread <=> %s) >= %s 
            ORDER BY embeddings_mxdbread <=> %s 
            LIMIT %s
        """
        params = (prompt_vector, 0.4, prompt_vector, 2)
        
        # Use execute_query to fetch results
        result = self.db_connector.execute_query(query, params, fetch_mode="all")
        
        return result
    
    def format_docs_2(context, source):
        formatted_docs = []
        # Iterate over each document in the source list
        for doc_content, doc_source in source:
            # Format each document's page content and metadata
            formatted_doc = f"context:\n```\n{doc_content}\n```\nsource:\n({doc_source})"
            formatted_docs.append(formatted_doc)
        # Join all formatted documents with two newlines
        return "\n\n".join(formatted_docs)

    @staticmethod
    def format_docs(docs):
        formatted_docs = []
        for doc in docs:
            # Format each document's page content and metadata
            formatted_doc = f"context:\n```\n{doc.page_content}\n```\nsource:\n({doc.metadata['source']})"
            formatted_docs.append(formatted_doc)
        # Join all formatted documents with two newlines
        return "\n\n".join(formatted_docs)

    @staticmethod
    def format_content(obj):
        return obj.content
    
    def create_history_aware_retriever(self):
            history_aware_retriever = self.history_aware_retriever_prompt | self.llm | StrOutputParser() | self.context_retriever
            return history_aware_retriever
     
    def create_chain(self):
        try:
            chain = (
                    {
                     "chat_history": itemgetter("chat_history"),
                     "context": self.history_aware_retriever | self.format_docs_2, 
                     "input": itemgetter("input")
                     }
                    | self.RAG_prompt
                    | self.llm
                    | StrOutputParser()
            )
            return chain
        except Exception as e:
            logging.error(f"Error creating Rag chain: {e}")
        
    def get_answer(self,question, session_id) -> str:
        with self.db_connector:
            self.chat_history = self.chat_history_service.retrieve_chat_history(session_id)
            logging.info(f"chat_history_retrieved = {self.chat_history}")
            # with self.db_connector:
            response = self.chain.invoke({"chat_history":self.chat_history, "input":question}, config={"callbacks": [langfuse_handler]})
            logging.info(f"""
                            chat_history after response
                            chat_history = {self.chat_history}, question = {question}, response = {response}
                            """)
            self.chat_history = self.chat_history_service.update_chat_history(self.chat_history, question, response)
            self.chat_history_service.store_chat_history(self.chat_history, session_id)
            return response