import os
import logging
import urllib.request
from PIL import Image
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Distance, VectorParams


class EmbeddingManager:
    def __init__(self, qdrant_client, model_path="clip-ViT-B-32"):
        self.qdrant_client = qdrant_client
        self.model = SentenceTransformer(model_path)
        logging.basicConfig(level=logging.INFO)  # Configure logging at the class level if not configured globally

    def get_file(self, image_path):
        #os.makedirs("./images", exist_ok=True)
        target_path = os.path.join("images", image_path)
        if not os.path.exists(target_path):
            logging.error(f"Invalid path {target_path}")
            return None
        return target_path

    def upsert_to_db(self, points):
        if points:  # Check if points list is not empty
            self.qdrant_client.upsert(
                collection_name="images_new",
                points=[
                    rest.PointStruct(
                        id=point['id'],
                        vector=point['vector'].tolist(),
                        payload=point['payload']
                    )
                    for point in points
                ]
            )
            logging.info(f"{len(points)} images encoded & upserted.")
    
    def create_collection(self, collection_name):
        self.qdrant_client.recreate_collection(collection_name=collection_name,
                                    vectors_config = {
                                        "image": VectorParams(size = 512, 
                                                              distance = Distance.COSINE ) 
                                        })
        logging.info(f"Collection {collection_name} created ") 

    def process_and_upload_images(self, data):
        self.create_collection(collection_name="images_new")
        
        points = []
        total_images =  len(data['img_file'])  
        for i in range(total_images):
            #img = self.get_file(data[i])
            img = os.path.join("images", data['img_file'][i])
            
            if img:
                try:
                    image = Image.open(img)
                    embedding = self.model.encode(image).tolist()
                    
                    points.append({
                        "id": i,
                        "vector": embedding,
                        "payload": {"url": data['image_urls'][i], "name": data['img_file'][i]}
                    })
                except Exception as e:
                    logging.error(f"Error processing image {img}: {e}")

            if (i + 1) % 1000 == 0 or i == total_images - 1:
                self.upsert_to_db(points)
                points = []

        logging.info("All embeddings upserted to vector database.")