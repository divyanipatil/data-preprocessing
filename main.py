from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
import uvicorn
from preprocessing import TextPreprocessor
from augmentation import TextAugmenter
from starlette.middleware.sessions import SessionMiddleware

app = FastAPI()

# Add session middleware
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")

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
            "original_text": request.session.get("original_text", ""),
            "processed_text": request.session.get("processed_text", "")
        }
    )


@app.post("/process", response_class=HTMLResponse)
async def process_file(
        request: Request,
        preprocessing: List[str] = Form(default=[]),
        augmentation: Optional[str] = Form(default=None),
        file: Optional[UploadFile] = File(default=None)
):
    # Get existing text from session
    current_text = request.session.get("original_text", "")

    if file:
        # If new file is uploaded, read it and store in session
        content = await file.read()
        text = content.decode()
        request.session["original_text"] = text
    elif current_text:
        # Use existing text from session if no new file is uploaded
        text = current_text
    else:
        # No file uploaded and no text in session
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "original_text": "",
                "processed_text": "Please upload a file first!"
            }
        )

    # First apply preprocessing
    processor = TextPreprocessor()
    processed_text = text

    for step in preprocessing:
        if step == "lowercase":
            processed_text = processor.to_lowercase(processed_text)
        elif step == "punctuation":
            processed_text = processor.remove_punctuation(processed_text)
        elif step == "lemmatize":
            processed_text = processor.lemmatize_text(processed_text)

    # Then apply augmentation if selected (only one at a time)
    if augmentation and augmentation != "none":
        augmenter = TextAugmenter()
        if augmentation == "word_swap":
            processed_text = augmenter.word_swap(processed_text)
        elif augmentation == "synonym":
            processed_text = augmenter.synonym_replacement(processed_text)

    # Store processed text in session
    request.session["processed_text"] = processed_text

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
