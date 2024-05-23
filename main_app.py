import os
import logging
import pandas as pd
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from embedding_manager import EmbeddingManager 
from app_interface import AppInterface
from qdrant_client.models import Distance, VectorParams


if __name__ == "__main__":
    # Load environment variables from the .env file
    load_dotenv(os.path.join(os.path.dirname(__file__), "qdrant.env"))

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Initialize the Qdrant client with environment variables
    qdrant_client = QdrantClient(
        url=os.getenv('QDRANT_DB_URL'),
        api_key=os.getenv('QDRANT_API_KEY'),
    )
    
    collection_name="text2img_search_collection"
    # Initialize and configure the Streamlit-based interface
    logging.info("Launching Text2Img Search Interface")
    try:
        app_interface = AppInterface(qdrant_client, collection_name)
        app_interface.launch()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        