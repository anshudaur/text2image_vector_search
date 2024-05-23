# Text to Image vector search using qdrant

## Data Exploration 

In the following notebook, we explore the images from the [data](data/images.tsv) created from the open_source dataset. Follow the steps in the notebook


[Data Exploration](notebooks/data_exploration.ipynb)


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

To create and upload embedding vectors in qdrant :

```
streamlit run main_qdrant.py 
```

To Launch text to image search application

```
streamlit run main_app.py 
```

4. Application is now active on :

```
http://localhost:8501/
```


### Query Evaluation

- Provide a set of example queries that effectively showcase the capabilities of your implemented system. These queries should yield relevant images, demonstrating the accuracy and efficiency of your text2image search solution.

[Red Car](images/red_car.jpg)


- Provide examples of queries that do not perform well, accompanied by explanations outlining the shortcomings of the system.

Examples of 

[Red Car](images/red_car.jpg)


- Suggest a method of quantitive evaluation of retrieval accuracy. (e.g. how to label dataset and prepare queries?)

Evaluation method for the system can be Recall@5, as getting results in top ...

To improve quantitative evaluation it would be useful to work with labelled data and benchmark multiple multimodal against the datasets to evaluate the performance of the model generating image embeddings/vectors. 

To label the dataset, we can use platforms like Scale AI etc to get the images labelled with the captions describing the contents of the images. The dataset then needs to be divided into reference and query set. In general the query set for retrieval should not overlap with the reference set to judge the retrieval performance of the system in real time.

Next we can run evaluation with multiple multi-modal approaches like CoCa , open-clip, Blip2 models.
Once the model with higher Recall@5 on the query set is obtained, this chosen model can be used to generate the new vectors.


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

```
Recall@5 = (#relevant images retrieved in top 5) / (Total relevant images)

```
 

System Architecture for Text-to-Image Search

Here’s a high-level block diagram for the architecture of a text-to-image search system that uses Recall@5 as an evaluation metric:


Data Ingestion: The system ingests images and their captions (which for now is only image id) into the database.
Feature Extraction: Extract features from each image using a pre-trained deep learning model, we used Sentence Transformer multi modal .
Indexing: Upload these features/embeddings in Qdrant for efficient similarity search.
Query Processing: Convert text queries into a vector using natural language processing techniques.
Search Engine: Perform vector similarity search to retrieve the top 5 closest images based on cosine similarity.
Evaluation Module: Calculate Recall@5 by comparing retrieved images against ground truth labels.


## Challenges

- Making open clip work with the local gpu and current setup, the models are bigger and embedding generation was slow. Hence, I had to retort to SentenceTransformer based clip model.

## Future Improvements

- Use labelled dataset with text/caption information
- Use Open Clip model to upsert text and image vectors to implement Cross modal retrieval system : image to text and text to image retrieval.

## References : 

- https://huggingface.co/sentence-transformers/clip-ViT-B-32