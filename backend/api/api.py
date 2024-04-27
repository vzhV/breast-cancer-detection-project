from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from models.input_models import MammogramAction, UltrasoundAction
from models.response_models import PredictionMGResponse, PredictionUSResponse
from service.mammogram import MammogramService
from service.ultrasound import UltrasoundService

app = FastAPI()

mammogram_service = MammogramService()
ultrasound_service = UltrasoundService()


@app.post("/mammogram", response_model=PredictionMGResponse)
async def mammogram(file: UploadFile = File(...), action: MammogramAction = Form(...)):
    """
        Endpoint for processing mammogram images. It supports actions: CLASSIFICATION, SEGMENTATION, ALL

        Args:
            file (UploadFile): The uploaded file containing a mammogram image.
            action (MammogramAction): The action to perform on the image, defined by an enum (Classification, Segmentation, or All).

        Returns:
            PredictionMGResponse: The response model containing the prediction results.

        Raises:
            HTTPException: If the uploaded file is not in an allowed format (.png, .jpg, .jpeg).
    """
    if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file format")
    prediction = mammogram_service.predict(file, action)
    return PredictionMGResponse.from_dto(prediction)


@app.post("/ultrasound", response_model=PredictionUSResponse)
async def ultrasound(file: UploadFile = File(...), action: UltrasoundAction = Form(...)):
    """
        Endpoint for processing ultrasound images. It supports actions: CLASSIFICATION, SEGMENTATION, CLASSIFICATION_OVERLAID

        Args:
            file (UploadFile): The uploaded file containing an ultrasound image.
            action (UltrasoundAction): The action to perform on the image, defined by an enum (Classification, Segmentation, Classification Overlaid).

        Returns:
            PredictionUSResponse: The response model containing the prediction results.

        Raises:
            HTTPException: If the uploaded file is not in an allowed format (.png, .jpg, .jpeg).
    """
    if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file format")
    prediction = ultrasound_service.predict(file, action)
    return PredictionUSResponse.from_dto(prediction)
