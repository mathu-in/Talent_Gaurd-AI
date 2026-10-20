from fastapi import FastAPI
import pandas as pd


from src.api.schema import Employee_Request
from src.components.Prediction_pipeline_14 import predict


app = FastAPI(
    title="Employee Request Prediction",
    version="1.0",
)

pipline = predict()

@app.get("/")
def home():
    return "Welcome to Employee Request Prediction"

@app.post("/predict")
def predict(employee_request: Employee_Request):
    employee_data = pd.DataFrame([employee_request.__dict__])
    prediction , probability = pipeline.predict(employee_data)

    if prediction == 1:
        result = 'Employee is likely to leave'
        confidence = probability[1]

    else:
        result = 'Employee is not likely to leave'
        confidence = probability[0]

    return {'Prediction' : int(prediction),
            'Confidence' : round(confidence*100,2),
            'Result' : result}