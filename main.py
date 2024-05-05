import os
import logging
from qdrant_client import QdrantClient
from dotenv import load_dotenv
# from embedding_manager import EmbeddingManager # Uncomment if needed
from AppInterface import AppInterface

if __name__ == "__main__":
    # Load environment variables from the .env file
    load_dotenv(os.path.join(os.path.dirname(__file__), "../qdrant.env"))

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Initialize the Qdrant client with environment variables
    qdrant_client = QdrantClient(
        url=os.getenv('QDRANT_DB_URL'),
        api_key=os.getenv('QDRANT_API_KEY'),
    )

    # Initialize and configure the Streamlit-based interface
    try:
        app_interface = AppInterface(qdrant_client)
        app_interface.launch()
    except Exception as e:
        logging.error(f"An error occurred: {e}")