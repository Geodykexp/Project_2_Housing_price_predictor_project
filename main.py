from typing import List
from pydantic import BaseModel, conint, confloat
import numpy as np
import pickle
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse

app = FastAPI(title="Housing Price Predictor")

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "Housing_price_predictor.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model pickle not found at {MODEL_PATH}. Run the training script to create it.")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


class InputModel(BaseModel):
    area: confloat(gt=0)
    bedrooms: conint(ge=0, le=4)
    bathrooms: conint(ge=0, le=4)
    stories: conint(ge=0, le=4)
    mainroad: conint(ge=0, le=4)
    guestroom: conint(ge=0, le=4)
    basement: conint(ge=0, le=4)
    hotwaterheating: conint(ge=0, le=4)
    airconditioning: conint(ge=0, le=4)
    parking: conint(ge=0, le=4)
    prefare: conint(ge=0, le=4)
    furnishingstatus: conint(ge=0, le=4)


FEATURES: List[str] = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "parking",
    "prefare",
    "furnishingstatus",
]


@app.get("/", response_class=HTMLResponse)
async def read_root():
    path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="index.html not found")
    with open(path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/predict")
async def predict(inp: InputModel):
    values = [getattr(inp, f) for f in FEATURES]
    arr = np.array([values], dtype=float)
    try:
        pred = model.predict(arr)
        return JSONResponse({"prediction": float(pred[0])})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/source/{filename}')
async def source_file(filename: str):
    allowed = {
        'Housing_price_predict.py': os.path.join(BASE_DIR, 'Housing_price_predict.py'),
        'Housing_price_predictor.py': os.path.join(BASE_DIR, 'Housing_price_predictor.py'),
    }
    if filename not in allowed:
        raise HTTPException(status_code=404, detail='File not allowed')
    path = allowed[filename]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail='File not found')
    with open(path, 'r', encoding='utf-8') as f:
        return PlainTextResponse(f.read())

