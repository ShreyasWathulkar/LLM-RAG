from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from models.qa_model import QaInputModel,QaOutputModel
from services.rag_service import RAGService
from services.chat_history_service import ChatHistoryService
from repository.postgres_connector import PostgresConnector
import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import json
import os
from models.rag_config import *

router = APIRouter(prefix="/v1")

load_dotenv()

llm = ChatGroq(
            model = "llama-3.1-70b-versatile",
            # model="llama-3.1-8b-instant",
            # model="llama3-70b-8192",
            # model = "llama3-8b-8192",
            temperature=0
        )

history_aware_retriever_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(RETRIEVER_SYSTEM_PROMPT_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(RETRIEVER_HUMAN_PROMPT_TEMPLATE)
        ])

RAG_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(RAG_SYSTEM_PROMPT_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(RAG_HUMAN_PROMPT_TEMPLATE)
        ])

embedding_model = SentenceTransformer(
            "mixedbread-ai/mxbai-embed-large-v1", device='cpu'
        )

db_connector = PostgresConnector()

chat_history_service = ChatHistoryService(db_connector)

rag_service = RAGService(
        llm=llm,
        history_aware_retriever_prompt=history_aware_retriever_prompt,
        RAG_prompt=RAG_prompt,
        embedding_model=embedding_model,
        db_connector=db_connector,
        chat_history_service=chat_history_service
    )

@router.post("/qa/answer")
async def answer(input_model: QaInputModel):
    try:
        output = rag_service.get_answer(input_model.question, input_model.session_id)
        logging.debug(f"output returned: {output} and output with the model {QaOutputModel(output=output).model_dump_json}")
        return QaOutputModel(output=output).model_dump()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return JSONResponse(content={"message": "Error occurred"}, status_code=500)
