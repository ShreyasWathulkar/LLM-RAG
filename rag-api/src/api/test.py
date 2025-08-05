import logging
from services.rag_service import RAGService

# Set up logging to display output
logging.basicConfig(level=logging.DEBUG)

def main():
    # Create an instance of RAG_service with a test question
    question = "write a short note on fcb and catalonia"
    rag_service = RAGService(question)

    # Invoke the model and print the response
    try:
        response = rag_service.get_answer()
        print(f"Model Response: {response}")
    except Exception as e:
        logging.error(f"An error occurred while invoking the model: {e}")

if __name__ == "__main__":
    main()
