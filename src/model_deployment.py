from fastapi import FastAPI
import mlflow
import mlflow.sklearn
import pandas as pd
from pydantic import BaseModel
from typing import List

# Crear una aplicación FastAPI
app = FastAPI()

# Cargar el modelo entrenado desde MLflow
model_uri = "runs:/<RUN_ID>/random_forest_model"  # Sustituye <RUN_ID> con el RUN_ID real
model = mlflow.sklearn.load_model(model_uri)

# Definir el modelo de entrada esperado
class InputData(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    # Agrega más características según lo que necesites para tu modelo

class InputDataList(BaseModel):
    data: List[InputData]

# Ruta de predicción
@app.post("/predict")
def predict(input_data: InputDataList):
    # Convertir los datos recibidos en un DataFrame
    input_dict = [data.dict() for data in input_data.data]
    df = pd.DataFrame(input_dict)

    # Hacer la predicción
    predictions = model.predict(df)

    # Retornar las predicciones
    return {"predictions": predictions.tolist()}
