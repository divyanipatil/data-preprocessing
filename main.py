from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List
import uvicorn
from preprocessing import TextPreprocessor

app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "original_text": "",
            "processed_text": ""
        }
    )


@app.post("/process", response_class=HTMLResponse)
async def process_file(
        request: Request,
        file: UploadFile = File(...),
        preprocessing: List[str] = Form(...)
):
    content = await file.read()
    text = content.decode()

    processor = TextPreprocessor()
    processed_text = text

    for step in preprocessing:
        if step == "lowercase":
            processed_text = processor.to_lowercase(processed_text)
        elif step == "punctuation":
            processed_text = processor.remove_punctuation(processed_text)
        elif step == "lemmatize":
            processed_text = processor.lemmatize_text(processed_text)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "original_text": text,
            "processed_text": processed_text
        }
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
