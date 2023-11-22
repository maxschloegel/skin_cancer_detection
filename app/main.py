from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel
import numpy as np

app = FastAPI()

# Pydantic model for the input data of the model_inference endpoint
class ImageInput(BaseModel):
    image: List[List[List[float]]]

# Pydantic model for the output data of the model_inference endpoint
class InferenceOutput(BaseModel):
    result: str  # Adjust the data type based on the actual output of your model

# Pydantic model for the output data of the model_info endpoint
class ModelInfoOutput(BaseModel):
    info: dict

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app!"}

# Model inference endpoint
@app.post("/model_inference")
async def model_inference(input_data: ImageInput):
    # Perform the model inference (replace this with your actual model inference logic)
    result = "benign"#model_inference_function(input_data.image)

    # Return the result in the specified Pydantic model
    return InferenceOutput(result=result)

# Model info endpoint
@app.get("/model_info")
async def model_info():
    # Generate some random values for demonstration purposes
    random_info = {"param1": np.random.random(), "param2": np.random.random()}

    # Return the random values in the specified Pydantic model
    return ModelInfoOutput(info=random_info)

