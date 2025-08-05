# from langchain_community.llms import Ollama
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_google_genai import (
#     ChatGoogleGenerativeAI,
#     HarmBlockThreshold,
#     HarmCategory,
# )
# from langchain_cohere import (
#     CohereEmbeddings,
#     CohereRerank,
#     # ChatCohere
# )
from langchain.schema import StrOutputParser
from sentence_transformers import SentenceTransformer
import psycopg2
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from operator import itemgetter
import streamlit as st
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_postgres import PGVector
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_groq import ChatGroq

from config_pg import *

import os
import json
from dotenv import load_dotenv

# Load .env file
load_dotenv()

PG_CONN_PARAMS = os.getenv('PG_CONN_PARAMS')
pg_conn_params = json.loads(PG_CONN_PARAMS)
groq_api_key = os.getenv('GROQ_API_KEY')
CONNECTION_STRING = os.getenv('CONNECTION_STRING')


class EduBotCreator:

    def __init__(self):
        self.system_prompt_template_1 = SYSTEM_PROMPT_TEMPLATE_1
        self.system_prompt_template_2 = SYSTEM_PROMPT_TEMPLATE_2
        self.human_prompt_template_1 = HUMAN_PROMPT_TEMPLATE_1
        self.human_prompt_template_2 = HUMAN_PROMPT_TEMPLATE_2
        self.search_kwargs = SEARCH_KWARGS
        self.embedder = EMBEDDER
        self.connection_string = CONNECTION_STRING
        # self.model_type = MODEL_TYPE
        self.temperature = TEMPERATURE
        self.collection_name = COLLECTION_NAME
        self.pg_conn_params = pg_conn_params

    def create_chat_prompt_1(self):
        chat_prompt_1 = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt_template_1),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(self.human_prompt_template_1)
        ])
        return chat_prompt_1

    def create_chat_prompt_2(self):

        chat_prompt_2 = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt_template_2),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(self.human_prompt_template_2)
        ])
        return chat_prompt_2

    def create_embedding_model_instance(self):
        embedding_model = SentenceTransformer(
            "mixedbread-ai/mxbai-embed-large-v1", device='cpu'
        )
        return embedding_model

    def get_embedding(self, user_query):

        response = self.embedding_model

        embedding = response.encode(user_query)

        # Converting the embedding to the pgvector and returning it
        return '[' + ','.join(map(str, embedding)) + ']'

    def own_retriever(self, user_query):
        conn = psycopg2.connect(**self.pg_conn_params)
        cursor = conn.cursor()

        prompt_vector = self.get_embedding(user_query)
        cursor.execute('SET max_parallel_workers_per_gather = 4')
        cursor.execute(
            'SELECT text,url '
            'FROM madhav_news_scroll11 WHERE 1 - (embeddings_mxdbread <=> %(prompt_vector)s) >= %(match_threshold)s '
            'ORDER BY embeddings_mxdbread <=> %(prompt_vector)s LIMIT %(match_cnt)s',
            {'prompt_vector': prompt_vector, 'match_threshold': 0.4, 'match_cnt': 2}
        )
        result = cursor.fetchall()
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

    @staticmethod
    def format_chat_history(messages):
        if not messages:
            return ""

        formatted = ""
        for message in messages:
            if message["role"] == "human":
                formatted += f"Human: {message['content']}\n"
            elif message["role"] == "assistant":
                formatted += f"Assistant: {message['content']}\n\n"
        return formatted.strip()

    def create_history_aware_retriever(self):
        try:
            history_aware_retriever = self.chat_prompt_1 | self.llm | StrOutputParser() | self.own_retriever
            return history_aware_retriever
        except Exception as e:
            st.error(f"error creating history aware retriever: {e}")

    def create_bot(self):
        try:
            rag_chain = (
                    {"chat_history": itemgetter("chat_history"),
                     "context": self.history_aware_retriever | self.format_docs_2, "input": itemgetter("input")}
                    | self.chat_prompt_2
                    | self.llm
                    | StrOutputParser()
            )
            return rag_chain
        except Exception as e:
            st.error(f"Error creating Rag chain: {e}")

    def load_llm(self):
        # llm = ChatGoogleGenerativeAI(model=self.model_type, temperature=self.temperature, safety_settings={
        #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        #     HarmCategory.HARM_CATEGORY_HARASSMENT : HarmBlockThreshold.BLOCK_NONE,
        #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT : HarmBlockThreshold.BLOCK_NONE,
        #     HarmCategory.HARM_CATEGORY_HATE_SPEECH : HarmBlockThreshold.BLOCK_NONE

        # })
        llm = ChatGroq(
            # model = "llama-3.1-70b-versatile",
            # model="llama-3.1-8b-instant",
            model="llama3-70b-8192",
            # model = "llama3-8b-8192",
            temperature=0
        )
        return llm

    def load_vectorstore(self):
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedder,
            model_kwargs={'trust_remote_code': True}
        )
        vectorstore = PGVector.from_existing_index(collection_name=self.collection_name, embedding=embeddings,
                                                   connection=self.connection_string, )
        return vectorstore

    def create_edubot(self):
        self.chat_prompt_1 = self.create_chat_prompt_1()
        self.chat_prompt_2 = self.create_chat_prompt_2()
        # self.vectorstore = self.load_vectorstore()
        self.embedding_model = self.create_embedding_model_instance()
        self.llm = self.load_llm()
        # self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs=self.search_kwargs)
        self.history_aware_retriever = self.create_history_aware_retriever()
        self.bot = self.create_bot()
        return self.bot