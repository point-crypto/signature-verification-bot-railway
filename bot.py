import os
import cv2
import json
import numpy as np
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    ConversationHandler,
    filters
)

# ================= CONFIG =================
TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("BOT_TOKEN not set")

REFERENCE, WAIT_TEST = range(2)

AUDIT_FILE = "audit_log.json"
REFERENCE_DIR = "signatures"
MATCH_THRESHOLD = 75

os.makedirs(REFERENCE_DIR, exist_ok=True)

# ================= SAFE JSON =================
def safe_load_json(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, ValueError):
        with open(path, "w") as f:
            json.dump([], f)
        return []

def safe_write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

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

    return [
        int(np.sum(img < 128)),
        int(len(contours)),
        float(np.mean(edges)),
        float(img.shape[1] / img.shape[0])
    ]

def ml_similarity(f1, f2):
    diff = np.abs(np.array(f1) - np.array(f2))
    return float((1 / (1 + np.mean(diff))) * 100)

# ================= AUDIT =================
def write_audit(entry):
    logs = safe_load_json(AUDIT_FILE)
    logs.append(entry)
    safe_write_json(AUDIT_FILE, logs)

# ================= BOT FLOW =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["ref_count"] = 0
    await update.message.reply_text(
        "✍️ *Signature Verification Bot*\n\n"
        "📌 Send reference signatures (multiple allowed)\n"
        "📌 When finished, type /verify\n\n"
        "/history – View past results\n"
        "/graph – Accuracy graph",
        parse_mode="Markdown"
    )
    return REFERENCE

# ================= SAVE REFERENCES =================
async def save_reference(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.from_user.id)
    user_dir = os.path.join(REFERENCE_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)

    photo = await update.message.photo[-1].get_file()
    count = len(os.listdir(user_dir)) + 1
    path = os.path.join(user_dir, f"ref{count}.jpg")

    await photo.download_to_drive(path)
    context.user_data["ref_count"] += 1

    await update.message.reply_text(
        f"✅ Reference {count} saved.\n"
        "Send more or type /verify"
    )
    return REFERENCE

# ================= VERIFY =================
async def verify(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get("ref_count", 0) == 0:
        await update.message.reply_text("❌ Send at least one reference first.")
        return REFERENCE

    await update.message.reply_text("📤 Now send TEST signature")
    return WAIT_TEST

# ================= TEST SIGNATURE =================
async def test_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.from_user.id)
    user_dir = os.path.join(REFERENCE_DIR, user_id)

    photo = await update.message.photo[-1].get_file()
    test_path = "test.jpg"
    await photo.download_to_drive(test_path)

    best_score = 0

    for file in os.listdir(user_dir):
        ref_img = preprocess(os.path.join(user_dir, file))
        test_img = preprocess(test_path)

        if ref_img is None or test_img is None:
            continue

        s = ssim(ref_img, test_img) * 100
        m = ml_similarity(
            extract_features(ref_img),
            extract_features(test_img)
        )

        final = (0.7 * s) + (0.3 * m)
        best_score = max(best_score, final)

    score = float(best_score)
    result = "MATCH ✅" if score >= MATCH_THRESHOLD else "MISMATCH ❌"
    confidence = "HIGH 🟢" if score >= 85 else "MEDIUM 🟡" if score >= 70 else "LOW 🔴"
    risk = float(100 - score)

    await update.message.reply_text(
        f"🔍 *Signature Result*\n\n"
        f"Score: `{score:.2f}%`\n"
        f"{result}\n"
        f"Confidence: {confidence}\n"
        f"Forgery Risk: `{risk:.2f}%`",
        parse_mode="Markdown"
    )

    write_audit({
        "time": datetime.now().isoformat(),
        "user_id": int(user_id),
        "final": round(score, 2),
        "confidence": confidence,
        "risk": round(risk, 2),
        "result": result
    })

    return ConversationHandler.END

# ================= HISTORY =================
async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    logs = safe_load_json(AUDIT_FILE)

    user_logs = [l for l in logs if l.get("user_id") == user_id][-5:]
    if not user_logs:
        await update.message.reply_text("No history available.")
        return

    msg = "📜 *Last 5 Verifications*\n\n"
    for l in user_logs:
        msg += f"{l['time']} | {l['final']}% | {l['result']}\n"

    await update.message.reply_text(msg, parse_mode="Markdown")

# ================= GRAPH =================
async def graph(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    logs = safe_load_json(AUDIT_FILE)

    scores = [l["final"] for l in logs if l.get("user_id") == user_id]
    if len(scores) < 2:
        await update.message.reply_text("Not enough data to generate graph.")
        return

    plt.plot(scores, marker="o")
    plt.title("Signature Accuracy Trend")
    plt.xlabel("Attempt")
    plt.ylabel("Score (%)")
    plt.grid(True)

    path = "accuracy.png"
    plt.savefig(path)
    plt.close()

    await update.message.reply_photo(photo=open(path, "rb"))

# ================= MAIN =================
def main():
    app = Application.builder().token(TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            REFERENCE: [
                MessageHandler(filters.PHOTO, save_reference),
                CommandHandler("verify", verify),
            ],
            WAIT_TEST: [
                MessageHandler(filters.PHOTO, test_image),
            ],
        },
        fallbacks=[]
    )

    app.add_handler(conv)
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CommandHandler("graph", graph))

    print("🤖 Bot running (polling mode)")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()

