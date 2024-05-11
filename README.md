# Text to Image vector search using qdrant

## Data Exploration 

In the following notebook, we explore the images from the [data](data/images.tsv) created from the open_source dataset. Follow the steps in the notebook

```
[Data Exploration](notebooks/data_exploration.ipynb)
```

## Start Application : 

Before starting the application, update your  api key and cluster url from qdrant cloud app in (qdrant_env)[qdrant.env] file

1. Set up environment :

```
conda create -n text2img python=3.10
pip install -r requirements.txt

```

2. Launch qdrant to upload images and embeddings

```
docker pull qdrant/qdrant
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

```

3. Streamlit application : 

```
streamlit run main.py
```

4. Application is now active on :

```
http://localhost:8501/
```


### Query Evaluation

- Provide a set of example queries that effectively showcase the capabilities of your implemented system. These queries should yield relevant images, demonstrating the accuracy and efficiency of your text2image search solution.

[Red Car](output_img/red_car.jpg)



- Provide examples of queries that do not perform well, accompanied by explanations outlining the shortcomings of the system.

Examples of 

[Red Car](output_img/red_car.jpg)

Since the images are not labelled, It is not possible to quantify the accuracy of the output.


- Suggest a method of quantitive evaluation of retrieval accuracy. (e.g. how to label dataset and prepare queries?)

Evaluation method for the system can be Recall@5, as getting results in top


## System Architecture

images/

## Evaluation measure : 

I recommend using Recall@5 as an evaluation metric for text-to-image search system to ensure that it effectively retrieves relevant images within the top 5 results. Here’s a structured approach to implement this, including a labeling strategy:

### Labeling Strategy

The demo supports a flagged button which allows for the images that should be labelled correctly , the example can be shown in flagged_outputs/

The selected flagged images can be passed to annotation pipeline with the correct labels.

### Evaluation Metric: Recall@5

Recall@5 measures the proportion of relevant images that appear in the top 5 results of the query output. It's particularly useful for assessing whether the most relevant images are being surfaced by your search system.

Calculation:

```math
Recall@5 = Number of relevant images retrieved in top 5 / Total relevant images
```
 

System Architecture for Text-to-Image Search

Here’s a high-level block diagram for the architecture of a text-to-image search system that uses Recall@5 as an evaluation metric:


Data Ingestion: The system ingests images and their annotations (which for now is only image id) into the database.
Feature Extraction: Extract features from each image using a pre-trained deep learning model.
Indexing: Index these features in Qdrant for efficient similarity search.
Query Processing: Convert text queries into a vector using natural language processing techniques.
Search Engine: Perform vector similarity search to retrieve the top 5 closest images based on cosine similarity.
Evaluation Module: Calculate Recall@5 by comparing retrieved images against ground truth labels.

## Challenges

- Making open clip work with the local gpu and current setup, the models are bigger and embedding generation was slow.

## Future Improvements

- Use dataset with text informations
- Use Open Clip model to upsert text and image vectors to implement retrieval and return recall scores for the search system/


## References : 

- https://huggingface.co/sentence-transformers/clip-ViT-B-32