import os
import cv2
import json
import numpy as np
from datetime import datetime
from flask import Flask, request

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)

from skimage.metrics import structural_similarity as ssim

# ================= CONFIG =================
TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # Railway public URL

if not TOKEN:
    raise RuntimeError("BOT_TOKEN not set")

REFERENCE, TEST = range(2)
AUDIT_FILE = "audit_log.json"

# ================= IMAGE UTILS =================
def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, bin_img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, w, h = cv2.boundingRect(np.vstack(cnts))
        img = img[y:y+h, x:x+w]
    return cv2.resize(img, (300, 150))

def extract_features(img):
    edges = cv2.Canny(img, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return np.array([
        np.sum(img < 128),
        len(cnts),
        np.mean(edges)
    ])

def ml_similarity(f1, f2):
    return float(1 / (1 + np.mean(np.abs(f1 - f2))) * 100)

# ================= BOT HANDLERS =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["refs"] = []
    await update.message.reply_text(
        "✍️ Signature Verification Bot\n\n"
        "Send REFERENCE signatures.\n"
        "Send /done when finished."
    )
    return REFERENCE

async def save_reference(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = await update.message.photo[-1].get_file()
    path = f"ref_{len(context.user_data['refs'])+1}.jpg"
    await photo.download_to_drive(path)
    context.user_data["refs"].append(path)
    await update.message.reply_text("✅ Reference saved")
    return REFERENCE

async def done_refs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📤 Send TEST signature")
    return TEST

async def test_signature(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = await update.message.photo[-1].get_file()
    test_path = "test.jpg"
    await photo.download_to_drive(test_path)

    test_img = preprocess(test_path)
    scores = []

    for ref in context.user_data["refs"]:
        ref_img = preprocess(ref)
        ssim_score = ssim(ref_img, test_img) * 100
        ml_score = ml_similarity(
            extract_features(ref_img),
            extract_features(test_img)
        )
        scores.append(0.7 * ssim_score + 0.3 * ml_score)

    final = max(scores) if scores else 0
    result = "MATCH ✅" if final >= 75 else "MISMATCH ❌"

    await update.message.reply_text(
        f"🔍 Score: {final:.2f}%\nResult: {result}"
    )
    return ConversationHandler.END

# ================= APP =================
flask_app = Flask(__name__)
application = Application.builder().token(TOKEN).build()

conv = ConversationHandler(
    entry_points=[CommandHandler("start", start)],
    states={
        REFERENCE: [
            MessageHandler(filters.PHOTO, save_reference),
            CommandHandler("done", done_refs),
        ],
        TEST: [MessageHandler(filters.PHOTO, test_signature)],
    },
    fallbacks=[]
)

application.add_handler(conv)

@flask_app.route("/webhook", methods=["POST"])
async def webhook():
    update = Update.de_json(request.get_json(force=True), application.bot)
    await application.process_update(update)
    return "OK"

@flask_app.route("/")
def home():
    return "Bot is running"

if __name__ == "__main__":
    application.run_webhook(
        listen="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
        webhook_url=f"{WEBHOOK_URL}/webhook"
    )

