import os
import json
import cv2
import numpy as np
from datetime import datetime
from skimage.metrics import structural_similarity as ssim

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters
)

# ================= CONFIG =================
TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("BOT_TOKEN environment variable not set")

AUDIT_FILE = "audit_log.json"

REFERENCE, TEST = range(2)

# ================= IMAGE PROCESSING =================
def preprocess(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(np.vstack(contours))
        img = img[y:y+h, x:x+w]

    return cv2.resize(img, (300, 150))

def extract_features(img):
    edges = cv2.Canny(img, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return np.array([
        float(np.sum(img < 128)),
        float(len(contours)),
        float(np.mean(edges)),
        float(img.shape[1] / img.shape[0])
    ])

def ml_similarity(f1, f2):
    diff = np.abs(f1 - f2)
    return float(1 / (1 + np.mean(diff)) * 100)

# ================= AUDIT =================
def write_audit(data):
    logs = []
    if os.path.exists(AUDIT_FILE):
        try:
            with open(AUDIT_FILE, "r") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []

    logs.append(data)
    with open(AUDIT_FILE, "w") as f:
        json.dump(logs, f, indent=4)

# ================= BOT COMMANDS =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        "✍️ *Signature Verification Bot*\n\n"
        "Send one or more *REFERENCE* signatures.\n"
        "After that, send the *TEST* signature.\n\n"
        "📌 Send /done when reference upload is finished.",
        parse_mode="Markdown"
    )
    context.user_data["refs"] = []
    return REFERENCE

async def save_reference(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = await update.message.photo[-1].get_file()
    idx = len(context.user_data["refs"]) + 1
    path = f"ref_{idx}.jpg"
    await photo.download_to_drive(path)

    context.user_data["refs"].append(path)
    await update.message.reply_text(f"✅ Reference {idx} saved.")
    return REFERENCE

async def done_references(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get("refs"):
        await update.message.reply_text("❌ Upload at least one reference signature.")
        return REFERENCE

    await update.message.reply_text("📤 Now send the TEST signature.")
    return TEST

async def test_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = await update.message.photo[-1].get_file()
    test_path = "test.jpg"
    await photo.download_to_drive(test_path)

    test_img = preprocess(test_path)
    if test_img is None:
        await update.message.reply_text("❌ Invalid test image.")
        return ConversationHandler.END

    scores = []
    for ref_path in context.user_data["refs"]:
        ref_img = preprocess(ref_path)
        if ref_img is None:
            continue

        ssim_score = float(ssim(ref_img, test_img) * 100)
        ml_score = ml_similarity(
            extract_features(ref_img),
            extract_features(test_img)
        )
        scores.append(0.7 * ssim_score + 0.3 * ml_score)

    final_score = max(scores) if scores else 0.0
    result = "MATCH ✅" if final_score >= 75 else "MISMATCH ❌"
    confidence = "HIGH 🟢" if final_score >= 85 else "LOW 🔴"

    report = (
        "🔍 *Signature Analysis*\n\n"
        f"Score : `{final_score:.2f}%`\n"
        f"Result : *{result}*\n"
        f"Confidence : {confidence}"
    )

    await update.message.reply_text(report, parse_mode="Markdown")

    write_audit({
        "time": datetime.now().isoformat(),
        "score": round(final_score, 2),
        "result": result
    })

    return ConversationHandler.END

# ================= MAIN =================
def main():
    app = Application.builder().token(TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            REFERENCE: [
                MessageHandler(filters.PHOTO, save_reference),
                CommandHandler("done", done_references),
            ],
            TEST: [MessageHandler(filters.PHOTO, test_image)],
        },
        fallbacks=[]
    )

    app.add_handler(conv)
    print("✅ Bot running correctly")
    app.run_polling()

if __name__ == "__main__":
    main()
