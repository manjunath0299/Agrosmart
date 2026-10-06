import os
import tempfile
from pathlib import Path

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException

from backend.disease_api_helper import predict_uploaded_image
from backend.disease_formatter import format_disease_result


app = FastAPI(
    title="Smart Agriculture Disease Detection API",
    version="1.0.0"
)


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

TELEGRAM_API_URL = (
    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    if TELEGRAM_BOT_TOKEN
    else ""
)


@app.get("/")
def root():
    return {
        "service": "Smart Agriculture Disease Detection",
        "status": "running",
        "model": "EfficientNetV2-B0",
        "endpoint": "/predict-disease"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict-disease")
async def predict_disease(file: UploadFile = File(...)):

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format."
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Empty image file."
        )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            temp_file.write(contents)
            temp_path = temp_file.name

        result = predict_uploaded_image(temp_path)

        result["farmer_message"] = format_disease_result(result)

        result["uploaded_file"] = file.filename

        return result

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/telegram-webhook")
async def telegram_webhook(update: dict):

    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="TELEGRAM_BOT_TOKEN is not configured."
        )

    message = update.get("message")

    if not message:
        return {"ok": True}

    chat = message.get("chat", {})
    chat_id = chat.get("id")

    if not chat_id:
        return {"ok": True}

    # Handle /start
    text = message.get("text", "")

    if text == "/start":

        await send_telegram_message(
            chat_id,
            "🌱 Welcome to Smart Agriculture Disease Detection!\n\n"
            "📸 Send me a clear photo of a plant leaf and "
            "I will analyze it using the AI disease detection model."
        )

        return {"ok": True}

    # Handle /help
    if text == "/help":

        await send_telegram_message(
            chat_id,
            "🌱 Smart Agriculture AI\n\n"
            "Send a clear plant-leaf image.\n\n"
            "Tips:\n"
            "• Good lighting\n"
            "• Keep the leaf in focus\n"
            "• Minimize background\n"
            "• Show the affected region clearly"
        )

        return {"ok": True}

    # Handle photo
    photos = message.get("photo")

    if photos:

        try:

            # Telegram provides several resolutions.
            # The last item is normally the largest.
            photo = photos[-1]

            file_id = photo.get("file_id")

            if not file_id:
                return {"ok": True}

            image_bytes = await download_telegram_file(file_id)

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            ) as temp_file:

                temp_file.write(image_bytes)
                temp_path = temp_file.name

            try:

                result = predict_uploaded_image(temp_path)

                farmer_message = format_disease_result(result)

                await send_telegram_message(
                    chat_id,
                    farmer_message
                )

            finally:

                if os.path.exists(temp_path):
                    os.remove(temp_path)

        except Exception as e:

            print(f"Telegram image processing error: {e}")

            await send_telegram_message(
                chat_id,
                "⚠️ Sorry, I could not process this image.\n\n"
                "Please try again with a clearer plant-leaf photo."
            )

        return {"ok": True}

    return {"ok": True}


async def send_telegram_message(
    chat_id,
    text
):

    url = f"{TELEGRAM_API_URL}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": text
    }

    async with httpx.AsyncClient(timeout=30) as client:

        response = await client.post(
            url,
            json=payload
        )

        response.raise_for_status()


async def download_telegram_file(
    file_id: str
):

    # First ask Telegram for the file path.
    get_file_url = f"{TELEGRAM_API_URL}/getFile"

    async with httpx.AsyncClient(timeout=30) as client:

        response = await client.get(
            get_file_url,
            params={
                "file_id": file_id
            }
        )

        response.raise_for_status()

        data = response.json()

        file_path = data["result"]["file_path"]

        # Download actual image.
        download_url = (
            f"https://api.telegram.org/file/"
            f"bot{TELEGRAM_BOT_TOKEN}/"
            f"{file_path}"
        )

        image_response = await client.get(
            download_url,
            timeout=180
        )

        image_response.raise_for_status()

        return image_response.content