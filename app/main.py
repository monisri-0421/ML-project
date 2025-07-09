from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from src.predict import predict_image
from starlette.responses import RedirectResponse
import shutil

app = FastAPI()

# Set up template directory
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    prediction = predict_image(contents)
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})

# Existing JSON API endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    result = predict_image(contents)
    return {"prediction": result}
