import io

from fastapi import FastAPI, File, UploadFile
from omegaconf import OmegaConf
import PIL.Image as Image
from pydantic import BaseModel

from skin_cancer_detection.utils import (load_model, get_label_for_prediction,
                                         apply_transforms)


app = FastAPI()

# Load model
tf, model = load_model()
model.eval()


class InferenceOutput(BaseModel):
    result: str
    certainty: float


class ModelInfoOutput(BaseModel):
    model_info: dict


# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app!"}


# Model inference endpoint
@app.post("/model_inference")
async def model_inference(file: UploadFile = File(...)):
    # load image
    contents = await file.read()
    im = Image.open(io.BytesIO(contents))

    # apply transforms and model
    im_t = apply_transforms(tf, im).to(model.device)
    predictions = model(im_t[None, :, :, :])

    result, certainty = get_label_for_prediction(predictions,
                                                 model.hparams["data"]["class_strings"])  # noqa: E501

    return InferenceOutput(result=result, certainty=certainty)


# Model info endpoint
@app.get("/model_info")
async def model_info():
    random_info = OmegaConf.to_container(model.hparams)
    return ModelInfoOutput(model_info=random_info)
