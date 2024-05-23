import os
import logging
import pandas as pd
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from embedding_manager import EmbeddingManager 
from app_interface import AppInterface
from qdrant_client.models import Distance, VectorParams

def collection_exists(qdrant_client, collection_name):
        """Check if a collection exists."""
        try:
            qdrant_client.get_collection(collection_name)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return False
        return True
    
def create_collection(qdrant_client, collection_name):
    qdrant_client.recreate_collection(collection_name=collection_name,
                                    vectors_config=VectorParams(size=512, distance=Distance.COSINE),)
    logging.info(f"Collection {collection_name} created ") 

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
    if not collection_exists(qdrant_client, collection_name):
        logging.info("Create Collection")
        create_collection(qdrant_client, collection_name=collection_name)

        # Load data from a TSV file
        data = pd.read_csv('data/images.csv')
        logging.info(f"Data Loaded: {data.shape}\n{data.head()}")
        logging.info(len(data['img_file']))
            
        # Initialize and use the EmbeddingManager
        try:
            embedding_manager = EmbeddingManager(qdrant_client)  # Initialize EmbeddingManager
            embedding_manager.process_and_upload_images(collection_name, data)  # Process and upload images
        except Exception as e:
            logging.error(f"Error processing and uploading images: {e} ")
        logging.info("Collection created")
    else:
        logging.info("Collection already created")
        