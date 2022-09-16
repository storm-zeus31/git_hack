from fastapi import FastAPI,File,UploadFile
import uvicorn
from fastapi.responses import Response
from starlette.responses import RedirectResponse
from classification import prediction, read_imagefile

val= []

app =FastAPI(title='Image Classifier')

@app.post("/classify/image")
async def classifyImage(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image should be jpg or png format!"
    image = read_imagefile(await file.read())
    predictionResult = prediction(image)

    return predictionResult

if __name__== "__main__":
    uvicorn.run(app, debug=True)