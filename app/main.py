from fastapi import FastAPI, UploadFile, File
from src.predict import predict_image

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    result = predict_image(contents)
    return {"prediction": result}

