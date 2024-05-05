import os
import gradio as gr
import logging
from PIL import Image, ImageDraw, ImageFont
import urllib.request
from sentence_transformers import SentenceTransformer
from qdrant_client.http import models

class AppInterface:
    def __init__(self, qdrant_client):
        self.qdrant_client = qdrant_client
        self.model = SentenceTransformer("clip-ViT-B-32")
        logging.basicConfig(level=logging.INFO)    

    def add_text_to_image(self, image, text):
        """Adds text overlay to an image."""
        draw = ImageDraw.Draw(image)
        
        # Use a truetype font if available, otherwise fall back to the default font
        try:
            font = ImageFont.truetype("data/Arial_Bold.ttf", 20)  # Specify path to a TrueType font and size
        except IOError:
            font = ImageFont.load_default()  # Load default font if custom font is unavailable
    
        # Calculate text width and height using the correct font
        textwidth, textheight = draw.textsize(text, font=font)
        
        # Position the text at the top center
        width, height = image.size
        x = (width - textwidth) / 2
        y = 10  # Margin from the top of the image
    
        # Draw the text on the image in white, add shadow or outline for better visibility if needed
        shadowcolor = "black"
        draw.text((x-1, y-1), text, font=font, fill=shadowcolor)
        draw.text((x+1, y-1), text, font=font, fill=shadowcolor)
        draw.text((x-1, y+1), text, font=font, fill=shadowcolor)
        draw.text((x+1, y+1), text, font=font, fill=shadowcolor)
        draw.text((x, y), text, font=font, fill="white")  # Text itself in white
    
        return image

    def process_text_to_image(self, query_text):
        images = []  # Ensure images is defined here to be used throughout
        scores = []  # Same with scores
        try:
            query_vector = self.model.encode(query_text).tolist()
            results = self.qdrant_client.search(
                collection_name="images",
                query_vector=query_vector,
                with_payload=True,
                limit=5
            )
            if not results:
                return [] #, []

            for hit in results:
                try:
                    
                    img_url = hit.payload['url']
                    image = Image.open(urllib.request.urlopen(img_url))
                    images.append(image)
                    score_text = f"Score: {hit.score:.2f}"
                    logging.info(f"score_text : {score_text} ")
                    #score = f"Score: {hit.score:.2f}"  # Formatting the score as a string
                    #image_with_text = self.add_text_to_image(image, score)
                    scores.append(hit.score)
                    #scores.append(score_text)
                    #formatted_scores = [[score] for score in scores]  # Wrap each score in a list
                    #scores = formatted_scores
                    
                except Exception as e:
                    logging.error(f"Error loading or processing image from URL {img_url}: {e}")
        except Exception as e:
            logging.error(f"Error processing query '{query_text}': {e}")
            return [] #, []  # Ensure all return paths give correct types

        # Now you can safely check the length of images
        return images #, scores  # Return just the images with text overlay, no need for separate scores list

    def launch(self):
        iface = gr.Interface(
            title="Text to Image Vector Search with Qdrant",
            description="Enter text to find matching images and view similarity scores",
            fn=self.process_text_to_image,
            inputs=gr.Textbox(label="Query Text"),
            outputs=[gr.Gallery(label="Relevant Images"),  
                     #gr.Dataframe(label="Similarity Scores"),
                     #gr.Label(label="Similarity Scores")
                    ],
        )
        iface.launch()