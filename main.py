from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
import uvicorn
from preprocessing import TextPreprocessor
from augmentation import TextAugmenter
from image_preprocessing import ImagePreprocessor
from image_augmentation import ImageAugmenter
from starlette.middleware.sessions import SessionMiddleware
import PIL.Image as Image
from io import BytesIO

app = FastAPI()

# Add session middleware with explicit max_age
app.add_middleware(
    SessionMiddleware,
    secret_key="your-secret-key",
    max_age=3600,  # 1 hour in seconds
    same_site="lax",
    session_cookie="session"
)

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
            "processed_text": "",
            "original_image": "",
            "processed_image": ""
        }
    )


@app.post("/process_text")
async def process_text(
        request: Request,
        preprocessing: List[str] = Form(default=[]),
        augmentation: Optional[str] = Form(default=None),
        file: Optional[UploadFile] = File(default=None)
):
    if file:
        content = await file.read()
        text = content.decode()
    else:
        text = request.session.get("original_text", "")
        if not text:
            return JSONResponse({
                "error": "Please upload a file first!"
            })

    processor = TextPreprocessor()
    processed_text = text

    for step in preprocessing:
        if step == "lowercase":
            processed_text = processor.to_lowercase(processed_text)
        elif step == "punctuation":
            processed_text = processor.remove_punctuation(processed_text)
        elif step == "lemmatize":
            processed_text = processor.lemmatize_text(processed_text)

    if augmentation and augmentation != "none":
        augmenter = TextAugmenter()
        if augmentation == "word_swap":
            processed_text = augmenter.word_swap(processed_text)
        elif augmentation == "synonym":
            processed_text = augmenter.synonym_replacement(processed_text)

    request.session["original_text"] = text
    request.session["processed_text"] = processed_text

    return JSONResponse({
        "original_text": text,
        "processed_text": processed_text
    })


@app.post("/process_image")
async def process_image(
        request: Request,
        preprocessing: List[str] = Form(default=[]),
        augmentation: Optional[str] = Form(default=None),
        file: Optional[UploadFile] = File(default=None)
):
    try:
        if file:
            content = await file.read()
            image = Image.open(BytesIO(content))
            if image.mode != "RGB":
                image = image.convert("RGB")
        else:
            return JSONResponse({
                "error": "Please upload an image first!"
            })

        processor = ImagePreprocessor()
        processed_image = image.copy()

        for step in preprocessing:
            if step == "normalize":
                processed_image = processor.normalize_standard(processed_image)
            elif step == "resize":
                processed_image = processor.resize_224(processed_image)
            elif step == "grayscale":
                processed_image = processor.grayscale(processed_image)

        if augmentation and augmentation != "none":
            augmenter = ImageAugmenter()
            if augmentation == "flip":
                processed_image = augmenter.horizontal_flip(processed_image)
            elif augmentation == "rotate":
                processed_image = augmenter.rotate(processed_image)
            elif augmentation == "color":
                processed_image = augmenter.color_jitter(processed_image)

        # Resize images before base64 encoding to reduce size
        if image.size[0] > 800 or image.size[1] > 800:
            aspect_ratio = image.size[1] / image.size[0]
            new_width = 800
            new_height = int(new_width * aspect_ratio)
            image = image.resize((new_width, new_height), Image.LANCZOS)

        if processed_image.size[0] > 800 or processed_image.size[1] > 800:
            aspect_ratio = processed_image.size[1] / processed_image.size[0]
            new_width = 800
            new_height = int(new_width * aspect_ratio)
            processed_image = processed_image.resize((new_width, new_height), Image.LANCZOS)

        original_image_b64 = processor.to_base64(image)
        processed_image_b64 = processor.to_base64(processed_image)

        return JSONResponse({
            "original_image": original_image_b64,
            "processed_image": processed_image_b64
        })

    except Exception as e:
        return JSONResponse({
            "error": str(e)
        })


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
