from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile

from .inference import CaloriePredictor


predictor: CaloriePredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global predictor
    predictor = CaloriePredictor()
    yield


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    def read_root() -> dict[str, str]:
        return {"status": "API is running"}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> dict:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        result = predictor.predict(image_bytes)
        return {
            "food_name": result.get("food_name"),
            "calories": result.get("calories"),
            "confidence": result.get("confidence"),
        }

    return app


app = create_app()
