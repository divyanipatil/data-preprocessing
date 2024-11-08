from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
import uvicorn

from audio.audio_augmentation import AudioAugmenter
from audio.audio_preprocessing import AudioPreprocessor
from image.image_augmentation import ImageAugmenter
from image.image_preprocessing import ImagePreprocessor
from text.text_augmentation import TextAugmenter
from text.text_preprocessing import TextPreprocessor
from model.model_augmentation import ModelAugmenter
from model.model_preprocessing import ModelPreprocessor
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
            "processed_image": "",
            "original_audio": "",
            "processed_audio": ""
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


@app.post("/process_audio")
async def process_audio(
        request: Request,
        preprocessing: List[str] = Form(default=[]),
        augmentation: Optional[str] = Form(default=None),
        file: Optional[UploadFile] = File(default=None)
):
    try:
        if not file:
            return JSONResponse({
                "error": "Please upload an audio file first!"
            })

        # Check file extension
        if not file.filename.lower().endswith(('.mp3', '.wav')):
            return JSONResponse({
                "error": "Only .mp3 and .wav files are supported!"
            })

        content = await file.read()
        processor = AudioPreprocessor()

        try:
            waveform, sample_rate = processor.load_audio(content)
        except Exception as e:
            return JSONResponse({
                "error": f"Failed to load audio file: {str(e)}"
            })

        processed_waveform = waveform.clone()

        # Apply preprocessing
        for step in preprocessing:
            try:
                if step == "noise":
                    processed_waveform = processor.add_noise(processed_waveform)
                elif step == "lowpass":
                    processed_waveform = processor.apply_low_pass_filter(processed_waveform)
                elif step == "speed":
                    processed_waveform = processor.change_speed(processed_waveform)
            except Exception as e:
                print(f"Warning: Failed to apply {step}: {str(e)}")

        # Apply augmentation
        if augmentation and augmentation != "none":
            try:
                augmenter = AudioAugmenter()
                if augmentation == "timeshift":
                    processed_waveform = augmenter.time_shift(processed_waveform)
                elif augmentation == "pitch":
                    processed_waveform = augmenter.pitch_shift(processed_waveform)
            except Exception as e:
                print(f"Warning: Failed to apply augmentation: {str(e)}")

        # Convert to base64 for web playback
        try:
            original_audio_b64 = processor.to_base64(waveform, sample_rate)
            processed_audio_b64 = processor.to_base64(processed_waveform, sample_rate)
        except Exception as e:
            return JSONResponse({
                "error": f"Failed to convert audio for playback: {str(e)}"
            })

        return JSONResponse({
            "original_audio": original_audio_b64,
            "processed_audio": processed_audio_b64
        })

    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return JSONResponse({
            "error": f"Failed to process audio: {str(e)}"
        })


@app.post("/process_model")
async def process_model(
        request: Request,
        preprocessing: List[str] = Form(default=[]),
        augmentation: Optional[str] = Form(default=None),
        file: Optional[UploadFile] = File(default=None)
):
    try:
        if not file:
            return JSONResponse({
                "error": "Please upload a model file!"
            })

        if not file.filename.lower().endswith('.obj'):
            return JSONResponse({
                "error": "Only .obj files are supported!"
            })

        content = await file.read()
        processor = ModelPreprocessor()

        try:
            mesh = processor.load_obj(content)
            print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
        except Exception as e:
            print(f"Failed to load mesh: {str(e)}")
            return JSONResponse({
                "error": f"Failed to load mesh: {str(e)}"
            })

        # Make a copy for processing
        processed_mesh = mesh.copy()

        # Apply preprocessing
        for step in preprocessing:
            try:
                if step == "normalize":
                    processed_mesh = processor.normalize(processed_mesh)
                    print(f"Applied normalization")
                elif step == "center":
                    processed_mesh = processor.center_model(processed_mesh)
                    print(f"Applied smoothing")
            except Exception as e:
                print(f"Warning: Failed to apply {step}: {str(e)}")

        # Apply augmentation
        if augmentation and augmentation != "none":
            try:
                augmenter = ModelAugmenter()
                if augmentation == "rotate":
                    processed_mesh = augmenter.rotate(processed_mesh)
                    print("Applied rotation")
                elif augmentation == "scale":
                    processed_mesh = augmenter.scale_random(processed_mesh)
                    print("Applied scaling")
            except Exception as e:
                print(f"Warning: Failed to apply augmentation: {str(e)}")

        # Convert to JSON format for three.js
        try:
            original_model = processor.to_json(mesh)
            processed_model = processor.to_json(processed_mesh)
            print("Successfully converted models to JSON")
        except Exception as e:
            print(f"Failed to convert model for display: {str(e)}")
            return JSONResponse({
                "error": f"Failed to convert model for display: {str(e)}"
            })

        return JSONResponse({
            "original_model": original_model,
            "processed_model": processed_model
        })

    except Exception as e:
        print(f"Error processing model: {str(e)}")
        return JSONResponse({
            "error": f"Failed to process model: {str(e)}"
        })


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
