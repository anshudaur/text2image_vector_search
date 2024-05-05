import os
import logging
from qdrant_client import QdrantClient
import pandas as pd
from dotenv import load_dotenv
from embedding_manager import EmbeddingManager
from AppInterface import AppInterface

if __name__ == "__main__":
    # Load environment variables from the .env file
    load_dotenv(os.path.join(os.path.dirname(__file__), "../qdrant.env"))

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    try:
        # Initialize the Qdrant client with environment variables
        qdrant_client = QdrantClient(
            url=os.getenv('QDRANT_DB_URL'),
            api_key=os.getenv('QDRANT_API_KEY'),
        )

        # Load data from a TSV file
        data = pd.read_csv('data/images.tsv', sep='\t', header=None).reset_index(drop=True)
        logging.info(f"Data Loaded: {data.shape}")

        # Initialize and use the EmbeddingManager
        embedding_manager = EmbeddingManager(qdrant_client)
        embedding_manager.process_and_upload_images(data)

        # Initialize and launch the application interface
        app_interface = AppInterface(qdrant_client)
        app_interface.launch()

    except Exception as e:
        logging.error(f"An error occurred: {e}")
