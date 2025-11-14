# from flask import Blueprint, render_template, redirect
# import cv2
# import numpy as np
# from PIL import Image
# import os

# train_bp = Blueprint('train', __name__, template_folder='../templates')

# dataset_path = "dataset"
# recognizer_path = "recognizer"
# os.makedirs(recognizer_path, exist_ok=True)

# @train_bp.route('/', methods=['GET', 'POST'])
# def train_page():
#     # Check if dataset exists
#     if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) == 0:
#         return "⚠️ No images found in dataset. Please register some users first."

#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     faces, ids = get_images_and_labels(dataset_path)

#     recognizer.train(faces, np.array(ids))
#     recognizer.save(os.path.join(recognizer_path, "trainingdata.yml"))
#     print(f"✅ Training completed on {len(ids)} samples.")
#     return render_template('train_success.html', count=len(ids))


# def get_images_and_labels(path):
#     image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))]
#     faces = []
#     ids = []

#     for img_path in image_paths:
#         gray_img = Image.open(img_path).convert('L')
#         img_np = np.array(gray_img, 'uint8')
#         user_id = int(os.path.split(img_path)[-1].split(".")[0])  # e.g., "1.1.jpg" → 1
#         faces.append(img_np)
#         ids.append(user_id)
#     return faces, ids





from flask import Blueprint, render_template
import cv2
import numpy as np
from PIL import Image
import os
import sqlite3

# === Blueprint setup ===
train_bp = Blueprint('train', __name__, template_folder='../templates')

# === Paths ===
dataset_path = "dataset"
recognizer_path = "recognizer"
recognizer_file = os.path.join(recognizer_path, "trainingdata.yml")

# ✅ Absolute database path
DB_PATH = r"C:\Users\subod\OneDrive\Desktop\careSync\Face Detection\backend\database.db"

# Ensure recognizer directory exists
os.makedirs(recognizer_path, exist_ok=True)

# === Database helper ===
def get_db_connection():
    return sqlite3.connect(DB_PATH)

# === Route ===
@train_bp.route('/', methods=['GET', 'POST'])
def train_page():
    """Train the face recognizer from dataset images."""
    if not os.path.exists(dataset_path) or len(os.listdir(dataset_path)) == 0:
        return "⚠️ No images found in dataset. Please register some users first."

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = get_images_and_labels(dataset_path)

    if not faces:
        return "⚠️ No valid face images found for training."

    recognizer.train(faces, np.array(ids))
    recognizer.save(recognizer_file)

    print(f"✅ Training completed on {len(ids)} samples.")

    # Optional: Log training event to database
    try:
        with get_db_connection() as conn:
            conn.execute("""
                INSERT INTO training_log (model, sample_count, status)
                VALUES (?, ?, ?)
            """, ("LBPH", len(ids), "completed"))
            conn.commit()
    except Exception as e:
        print("⚠️ Failed to log training event:", e)

    return render_template('train_success.html', count=len(ids))

# === Helper ===
def get_images_and_labels(path):
    """Load face images and extract labels from filenames."""
    image_paths = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    faces = []
    ids = []

    for img_path in image_paths:
        try:
            gray_img = Image.open(img_path).convert('L')
            img_np = np.array(gray_img, 'uint8')
            user_id = int(os.path.split(img_path)[-1].split(".")[0])  # e.g., "1.1.jpg" → 1
            faces.append(img_np)
            ids.append(user_id)
        except Exception as e:
            print(f"⚠️ Skipping {img_path}: {e}")

    return faces, ids