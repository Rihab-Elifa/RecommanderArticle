import pandas as pd
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

class Product(BaseModel):
    id: str
    name: str
    description: str
    price: float

class History(BaseModel):
    id: str

class RequestBody(BaseModel):
    products: List[Product]
    history: List[History]

class ResponseBody(BaseModel):
    recommended_products: List[Product]

# Function to recommend articles based on similarity matrix
def recommend_articles(products,x):
    return [products[i] for i in x.argsort()]

@app.post("/recommend_products")
def recommend_products(request: RequestBody):
    # Access the products and history from the request body
    products = request.products
    history = request.history
    articles = [product.name + ' ' + product.description + ' ' + str(product.price) for product in products]

    uni_tfidf = text.TfidfVectorizer(stop_words="english")
    uni_matrix = uni_tfidf.fit_transform(articles)
    uni_sim = cosine_similarity(uni_matrix)

    # Get the history IDs
    history_ids = [item.id for item in history]

    # Find the index of the history IDs in the 'products' list
    history_indices = [i for i, product in enumerate(products) if str(product.id) in history_ids]

    # Calculate recommendation scores based on the history indices
    recommendation_scores = uni_sim[history_indices].sum(axis=0)

    # Sort the recommendation scores in descending order
    sorted_indices = recommendation_scores.argsort()[::-1]

    # Retrieve the recommended products based on the sorted indices
    recommended_products = recommend_articles(products, sorted_indices)
    for product, score in zip(recommended_products, recommendation_scores[sorted_indices]):
        print(f"Product: {product}, Score: {score}")
    # Return the recommended products as the response
    return ResponseBody(recommended_products=recommended_products)