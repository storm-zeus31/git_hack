from fastapi import FastAPI,File,UploadFile
from flask import jsonify
import tensorflow as tf
import uvicorn
from fastapi.responses import Response
import uuid
from starlette.responses import RedirectResponse



from classification import prediction, read_imagefile

app =FastAPI(title='Image Classifier')

@app.post("/classify/image")
async def classifyImage(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    predictionResult = prediction(image)

    return predictionResult

if __name__== "__main__":
    uvicorn.run(app, debug=True)