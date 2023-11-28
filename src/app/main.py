import io

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel
import PIL.Image as Image
import numpy as np
import torch.nn.functional as F
from torchvision import transforms


from skin_cancer_detection.utils import (load_model, get_label_for_prediction
                                         apply_transforms)


app = FastAPI()

# Pydantic model for the output data of the model_inference endpoint
class InferenceOutput(BaseModel):
    result: str  # Adjust the data type based on the actual output of your model
    certainty: float

# Pydantic model for the output data of the model_info endpoint
class ModelInfoOutput(BaseModel):
    info: dict

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app!"}

# Model inference endpoint
#async def model_inference(input_data: ImageInput):
@app.post("/model_inference")
async def model_inference(file: UploadFile = File(...)):
    # Perform the model inference (replace this with your actual model inference logic)

    contents = await file.read()

    im = Image.open(io.BytesIO(contents))
    im_arr = np.array(im)

    tf, model = load_model()
    model.eval()
    im_t = apply_transforms(tf, im).to(model.device)
    predictions = model(im_t[None, :, :, :])
    result, certainty = get_label_for_prediction(predictions,
                                                 model.hparams["class_strings"])

    # Return the result in the specified Pydantic model
    return InferenceOutput(result=result, certainty=certainty)

# Model info endpoint
@app.get("/model_info")
async def model_info():
    # Generate some random values for demonstration purposes
    random_info = {"param1": np.random.random(), "param2": np.random.random()}

    # Return the random values in the specified Pydantic model
    return ModelInfoOutput(info=random_info)

