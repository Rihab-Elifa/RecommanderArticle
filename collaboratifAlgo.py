import numpy as np
import pandas as pd

import sklearn
from sklearn.decomposition import TruncatedSVD
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI

app = FastAPI()

class ratings(BaseModel):
    UserId: str
    ProductId: str
    Rating: int
class RequestBody(BaseModel):
    amazon_ratings: List[ratings]

@app.post("/recommend_products")
def recommend_products(request: RequestBody):
    amazon_ratings=request.amazon_ratings
    #Utility Matrix based on products sold and user reviews
    ratings_utility_matrix = amazon_ratings.pivot_table(values='Rating', index='UserId', columns='ProductId', fill_value=0)
    ratings_utility_matrix.head()
    X = ratings_utility_matrix.T
    X1 = X
    #Decomposing the Matrix
    SVD = TruncatedSVD(n_components=10)
    decomposed_matrix = SVD.fit_transform(X)
    decomposed_matrix.shape
    #Correlation Matrix
    correlation_matrix = np.corrcoef(decomposed_matrix)
    correlation_matrix.shape

    #Isolating Product ID # 6117036094 from the Correlation Matrix
    X.index[1]
    

    product_names = list(X.index)
    product_ID = product_names.index(i)
    product_ID
    correlation_product_ID = correlation_matrix[product_ID]
    correlation_product_ID.shape
    #Recommending top 10 highly correlated products in sequence
    Recommend = list(X.index[correlation_product_ID > 0.90])

    return Recommend[0:9]