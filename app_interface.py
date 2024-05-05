import os
import logging
from PIL import Image, ImageDraw, ImageFont
import urllib.request
from sentence_transformers import SentenceTransformer
from qdrant_client.http import models
import streamlit as st

class AppInterface:
    def __init__(self, qdrant_client):
        self.qdrant_client = qdrant_client
        self.model = SentenceTransformer("clip-ViT-B-32")
        logging.basicConfig(level=logging.INFO)

    def process_text_to_image(self, query_text):
        images = []
        scores = []
        try:
            query_vector = self.model.encode(query_text).tolist()
            results = self.qdrant_client.search(
                collection_name="images",
                query_vector=query_vector,
                with_payload=True,
                limit=6
            )
            if not results:
                return [], []

            for hit in results:
                img_url = hit.payload['url']
                image = Image.open(urllib.request.urlopen(img_url))
                images.append(image)
                score_text = f"Score: {hit.score:.2f}"
                scores.append(score_text)
        except Exception as e:
            logging.error(f"Error processing query '{query_text}': {e}")
            st.error("Failed to process the query.")
        return images, scores

    def launch(self):
        st.title("Text to Image Vector Search with Qdrant")
        st.write("Enter text to find matching images and view similarity scores")
        query_text = st.text_input("Query Text")
        if st.button("Search"):
            images, scores = self.process_text_to_image(query_text)
            if images:
                columns_per_row = 3
                # Create rows of images, 3 per row
                for i in range(0, len(images), columns_per_row):
                    cols = st.columns(columns_per_row)  # Create columns for each row
                    for col, image, score in zip(cols, images[i:i + columns_per_row], scores[i:i + columns_per_row]):
                        col.image(image, caption=score, use_column_width=True)
            else:
                st.write("No results found.")
