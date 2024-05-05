import os
import logging
import urllib.request
from PIL import Image
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

class EmbeddingManager:
    def __init__(self, qdrant_client, model_path="clip-ViT-B-32"):
        self.qdrant_client = qdrant_client
        self.model = SentenceTransformer(model_path)
        logging.basicConfig(level=logging.INFO)  # Configure logging at the class level if not configured globally

    def download_file(self, url):
        os.makedirs("./images", exist_ok=True)
        basename = os.path.basename(url)
        target_path = os.path.join("./images", basename)
        if not os.path.exists(target_path):
            try:
                urllib.request.urlretrieve(url, target_path)
            except urllib.error.HTTPError as e:
                logging.error(f"Failed to download {url}: {e}")
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

    def process_and_upload_images(self, data):
        points = []
        total_images = min(15000, len(data))  # Process up to 15000 images or total in data
        for i in range(total_images):
            img = self.download_file(data.iloc[i, 1])
            if img:
                try:
                    image = Image.open(img)
                    embedding = self.model.encode(image)
                    points.append({
                        "id": i,
                        "vector": embedding,
                        "payload": {"url": data.iloc[i, 1]}
                    })
                except Exception as e:
                    logging.error(f"Error processing image {img}: {e}")

            if (i + 1) % 1000 == 0 or i == total_images - 1:
                self.upsert_to_db(points)
                points = []

        logging.info("All embeddings upserted to vector database.")