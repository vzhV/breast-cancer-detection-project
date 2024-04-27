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
    if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file format")
    prediction = mammogram_service.predict(file, action)
    return PredictionMGResponse.from_dto(prediction)


@app.post("/ultrasound", response_model=PredictionUSResponse)
async def ultrasound(file: UploadFile = File(...), action: UltrasoundAction = Form(...)):
    if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file format")
    prediction = ultrasound_service.predict(file, action)
    return PredictionUSResponse.from_dto(prediction)
