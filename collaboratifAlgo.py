import numpy as np
import pandas as pd

import sklearn
from sklearn.decomposition import TruncatedSVD
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI

app = FastAPI()

class ratings(BaseModel):
    NumR: int
    u: str
    a: str
   
class History(BaseModel):
    id: str
class RequestBody(BaseModel):
   ratings: List[ratings]
   hp: List[History]
   
class ResponseBody(BaseModel):
    recommended_products: List[str]

@app.post("/recommend")
def recommend_products(request: RequestBody):
    ratings=request.ratings
    hp=request.hp
    #Utility Matrix based on products sold and user reviews
    ratings_utility_matrix = ratings.pivot_table(values='NumR', index='u', columns='a', fill_value=0)
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
    #X.index[1]
    

    product_names = list(X.index)
    product_ID = product_names.index(hp.index[1])
    product_ID
    correlation_product_ID = correlation_matrix[product_ID]
    correlation_product_ID.shape
    #Recommending top 10 highly correlated products in sequence
    Recommend = list(X.index[correlation_product_ID > 0.90])

    return Recommend[0:9]