"""
Telegram interface for Smart Agriculture Disease Detection.

Flow:

Telegram
    ↓
Download image
    ↓
FastAPI
    ↓
Disease AI
    ↓
Farmer-friendly result
    ↓
Telegram
"""

import os
import tempfile
from pathlib import Path

import httpx
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN"
)


# ============================================================
# FASTAPI
# ============================================================

API_URL = (
    "http://127.0.0.1:8000/predict-disease"
)


# ============================================================
# VALIDATE TOKEN
# ============================================================

if not TELEGRAM_BOT_TOKEN:

    raise RuntimeError(
        "TELEGRAM_BOT_TOKEN was not found.\n"
        "Check C:\\smart_agriculture\\.env"
    )


# ============================================================
# START
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    message = (
        "🌱 *Smart Agriculture AI*\n\n"

        "Welcome! I can analyze plant leaf images "
        "for possible diseases.\n\n"

        "📷 Send a clear leaf photo to begin.\n\n"

        "💡 For better results:\n"
        "• Good lighting\n"
        "• Leaf clearly visible\n"
        "• Affected area in focus\n"
        "• Minimal background"
    )

    await update.message.reply_text(
        message,
        parse_mode="Markdown"
    )


# ============================================================
# HELP
# ============================================================

async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "📷 Send a clear plant leaf photo "
        "for AI disease screening."
    )


# ============================================================
# PHOTO HANDLER
# ============================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not update.message:
        return

    if not update.message.photo:
        return

    processing_message = await (
        update.message.reply_text(
            "🔄 Image received.\n"
            "📥 Downloading image..."
        )
    )

    temp_path = None

    try:

        # ====================================================
        # STEP 1 — GET TELEGRAM PHOTO
        # ====================================================

        print("\n" + "=" * 60)
        print("NEW TELEGRAM IMAGE")
        print("=" * 60)

        print(
            "Step 1/3: Getting Telegram image..."
        )

        # Highest resolution photo
        photo = update.message.photo[-1]

        print(
            f"Telegram file ID: {photo.file_id}"
        )

        print(
            f"Telegram image size: "
            f"{photo.width} x {photo.height}"
        )

        # ====================================================
        # STEP 2 — DOWNLOAD FROM TELEGRAM
        # ====================================================

        telegram_file = await photo.get_file(
            read_timeout=60,
            write_timeout=60,
            connect_timeout=60,
            pool_timeout=60
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as temp_file:

            temp_path = Path(
                temp_file.name
            )

        print(
            "Downloading image from Telegram..."
        )

        await telegram_file.download_to_drive(
            custom_path=str(temp_path),

            read_timeout=60,
            write_timeout=60,
            connect_timeout=60,
            pool_timeout=60
        )

        print(
            f"Image downloaded successfully: "
            f"{temp_path}"
        )

        await processing_message.edit_text(
            "🔄 Image received.\n"
            "🧠 Sending image to AI..."
        )

        # ====================================================
        # STEP 3 — SEND TO FASTAPI
        # ====================================================

        print(
            "Step 2/3: Sending image to FastAPI..."
        )

        image_bytes = temp_path.read_bytes()

        print(
            f"Image size: "
            f"{len(image_bytes) / 1024:.2f} KB"
        )

        timeout = httpx.Timeout(
            connect=30.0,
            read=180.0,
            write=30.0,
            pool=30.0
        )

        async with httpx.AsyncClient(
            timeout=timeout
        ) as client:

            response = await client.post(
                API_URL,

                files={
                    "file": (
                        "telegram_leaf.jpg",
                        image_bytes,
                        "image/jpeg"
                    )
                }
            )

        print(
            f"FastAPI response: "
            f"{response.status_code}"
        )

        # ====================================================
        # CHECK API
        # ====================================================

        if response.status_code != 200:

            print(
                "FastAPI returned an error:"
            )

            print(
                response.text
            )

            raise RuntimeError(
                "FastAPI returned HTTP "
                f"{response.status_code}"
            )

        # ====================================================
        # PARSE RESULT
        # ====================================================

        print(
            "Step 3/3: Reading AI result..."
        )

        result = response.json()

        print(
            "AI prediction received successfully."
        )

        print(
            f"Status: "
            f"{result.get('status')}"
        )

        prediction = result.get(
            "prediction",
            {}
        )

        print(
            f"Disease: "
            f"{prediction.get('disease')}"
        )

        print(
            f"Confidence: "
            f"{prediction.get('confidence')}%"
        )

        # ====================================================
        # FARMER MESSAGE
        # ====================================================

        farmer_message = result.get(
            "farmer_message"
        )

        if not farmer_message:

            farmer_message = (
                "⚠️ AI result received, "
                "but no formatted message was available."
            )

        # ====================================================
        # SEND TO TELEGRAM
        # ====================================================

        await processing_message.edit_text(
            farmer_message
        )

        print(
            "Result sent to Telegram successfully."
        )

        print("=" * 60)

    # ========================================================
    # TIMEOUT
    # ========================================================

    except (httpx.TimeoutException, TimeoutError) as error:

        print(
            "\nTIMEOUT ERROR:"
        )

        print(
            repr(error)
        )

        try:

            await processing_message.edit_text(
                "⏱️ The request timed out.\n\n"
                "Please try the image again."
            )

        except Exception:
            pass

    # ========================================================
    # GENERAL ERROR
    # ========================================================

    except Exception as error:

        print(
            "\nTELEGRAM PREDICTION ERROR:"
        )

        print(
            repr(error)
        )

        try:

            await processing_message.edit_text(
                "❌ I could not complete the analysis.\n\n"
                "Please try again with a clear "
                "plant leaf image."
            )

        except Exception:
            pass

    # ========================================================
    # CLEANUP
    # ========================================================

    finally:

        if temp_path is not None:

            try:

                temp_path.unlink(
                    missing_ok=True
                )

            except Exception:
                pass


# ============================================================
# TEXT HANDLER
# ============================================================

async def other_message_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "📷 Please send a clear plant leaf photo "
        "for disease detection."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SMART AGRICULTURE TELEGRAM BOT")
    print("=" * 60)

    print(
        "\nFastAPI endpoint:"
    )

    print(
        API_URL
    )

    print(
        "\nStarting Telegram bot..."
    )

    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    application.add_handler(
        CommandHandler(
            "start",
            start_command
        )
    )

    application.add_handler(
        CommandHandler(
            "help",
            help_command
        )
    )

    application.add_handler(
        MessageHandler(
            filters.PHOTO,
            photo_handler
        )
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            other_message_handler
        )
    )

    print(
        "\nBot is running."
    )

    print(
        "Press CTRL+C to stop."
    )

    print("=" * 60)

    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()