import numpy as np
import pandas as pd

import sklearn
from sklearn.decomposition import TruncatedSVD
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI

app = FastAPI()

class ratings(BaseModel):
    numR: int
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
    # Convert ratings_data to a Pandas DataFrame
    ratings_df = pd.DataFrame(ratings)
    print(ratings_df.columns)
    ratings_df.head()
    print(ratings_df)
   
    
    df = pd.DataFrame(ratings)

# Fonction pour extraire les valeurs
    def extract_value(cell):
     return cell[1]

# Appliquez la fonction aux colonnes correspondantes
    df['numR'] = df[0].apply(extract_value)
    df['u'] = df[1].apply(extract_value)
    df['a'] = df[2].apply(extract_value)

# Supprimez les colonnes originales
    df = df.drop([0, 1, 2], axis=1)

    # Pivot Table avec les colonnes 'u' et 'a' et 'numR' comme valeurs
    ratings_utility_matrix = df.pivot_table(values='numR', index='u', columns='a', fill_value=0)


    # ... Le reste de votre code ...
    ratings_utility_matrix.head()
    X = ratings_utility_matrix.T
    X1 = X
    #Decomposing the Matrix
    SVD = TruncatedSVD(n_components=2)
    decomposed_matrix = SVD.fit_transform(X)
    decomposed_matrix.shape
    #Correlation Matrix
    correlation_matrix = np.corrcoef(decomposed_matrix)
    correlation_matrix.shape

    #Isolating Product ID # 6117036094 from the Correlation Matrix
    #X.index[1]
    

    product_names = list(X.index)
    product_ID = product_names.index( '648929f523051d677d53f029')
    product_ID
    correlation_product_ID = correlation_matrix[product_ID]
    correlation_product_ID.shape
    #Recommending top 10 highly correlated products in sequence
    Recommend = list(X.index[correlation_product_ID > 0.90])

    response_body = ResponseBody(recommended_products=Recommend[0:9])

# Renvoyez cet objet ResponseBody en tant que réponse
    return response_body