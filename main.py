import os
import logging
import pandas as pd
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from embedding_manager import EmbeddingManager 
from app_interface import AppInterface

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
    
   # Load data from a TSV file
    data = pd.read_csv('data/images.csv', header=None).reset_index(drop=True)  # Corrected reset_index
    logging.info(f"Data Loaded: {data.shape}\n{data.head()}")

    # Initialize and use the EmbeddingManager
    try:
        embedding_manager = EmbeddingManager(qdrant_client)  # Initialize EmbeddingManager
        embedding_manager.process_and_upload_images(data)  # Process and upload images
    except Exception as e:
        logging.error(f"Error processing and uploading images: {e}")

    # Initialize and configure the Streamlit-based interface
    try:
        app_interface = AppInterface(qdrant_client)
        app_interface.launch()
    except Exception as e:
        logging.error(f"An error occurred: {e}")