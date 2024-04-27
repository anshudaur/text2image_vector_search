# Machine Learning Engineer Test Task: Text2Image Search

## Objective
This test task aims to evaluate your grasp of fundamental machine learning and development concepts.
Your task involves working with an image dataset to develop a system capable of searching for similar images based on textual queries.

## Dataset
Please choose whatever image dataset you prefer, for example:

- crawl some e-commerce website
- find something on Kaggle
- use prepared `Advertisement Image Dataset`

```
#!/bin/bash

wget https://storage.googleapis.com/ads-dataset/subfolder-0.zip
wget https://storage.googleapis.com/ads-dataset/subfolder-1.zip
```

## Task Requirements

### Data Exploration
- Provide summarized report of the dataset contents

### Text2Image Search Implementation
- Implement a service, capable of retrieving relevant images based on the text input provided.
  - For example: Text query "black cat" should return a picture of a black cat if it is present in the dataset

Tips:
- Use vector search with Qdrant
- Provide solution in a form of FastAPI service or streamlit application

### Query Evaluation

- Provide a set of example queries that effectively showcase the capabilities of your implemented system. These queries should yield relevant images, demonstrating the accuracy and efficiency of your text2image search solution.
- Provide examples of queries that do not perform well, accompanied by explanations outlining the shortcomings of the system.
- Suggest a method of quantitive evaluation of retrieval accuracy. (e.g. how to label dataset and prepare queries?)

## Submission Guidelines
- Upload your solution to your GitHub account, accompanied by clear instructions for running the text2image search system.
- Include a `README.md` describing the system's architecture and techniques employed, challenges encountered during the implementation, and potential avenues for improvement.
