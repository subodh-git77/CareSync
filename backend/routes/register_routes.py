from flask import Blueprint, render_template, request, jsonify, flash
from backend.email_utils import send_appointment_email
from werkzeug.security import generate_password_hash
import os
import sqlite3
import cv2
import base64
import numpy as np

# === Blueprint setup ===
register_bp = Blueprint("register", __name__, template_folder="../templates")

# === Base paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RECOGNIZER_DIR = os.path.join(BASE_DIR, "recognizer")
RECOGNIZER_PATH = os.path.join(RECOGNIZER_DIR, "trainingdata.yml")
DB_PATH = os.path.join(BASE_DIR, "database.db")

# === Ensure directories exist ===
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(RECOGNIZER_DIR, exist_ok=True)

# === Face detector ===
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# === Constants ===
MAX_SAMPLES_PER_USER = 40


# -------------------------
# DATABASE HELPERS
# -------------------------
def get_db_connection():
    """Connect to main database."""
    return sqlite3.connect(DB_PATH)


def insert_or_update(user_id, name, age, gender, email, password_hash, problem, appointment):
    """Insert or update user details."""
    conn = get_db_connection()
    cursor = conn.execute(
        "SELECT id FROM users WHERE id=? OR email=?",
        (user_id, email),
    )
    existing = cursor.fetchone()

    if existing:
        conn.execute(
            """
            UPDATE users
            SET name=?, age=?, gender=?, email=?, password_hash=?, problem=?, appointment_date=?
            WHERE id=?
            """,
            (name, age, gender, email, password_hash, problem, appointment, user_id),
        )
        print(f"Updated record for ID {user_id}")
    else:
        conn.execute(
            """
            INSERT INTO users (id, name, age, gender, email, password_hash, problem, appointment_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, name, age, gender, email, password_hash, problem, appointment),
        )
        print(f"Inserted new record for ID {user_id}")

    conn.commit()
    conn.close()


# -------------------------
# IMAGE HELPERS
# -------------------------
def decode_image(img_data: str):
    """Decode base64-encoded image into OpenCV BGR image."""
    try:
        img_bytes = base64.b64decode(img_data.split(",")[1])
        img_np = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    except Exception as e:
        print("Failed to decode image:", e)
        return None


def extract_face(image, size=(200, 200)):
    """Detect largest face and return preprocessed grayscale tile."""
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(90, 90))

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    face = gray[y:y + h, x:x + w]
    face = cv2.resize(face, size, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face = clahe.apply(face)

    return face


def train_recognizer():
    """Train LBPH recognizer from dataset/<user_id>.<sample>.jpg files."""
    face_samples = []
    ids = []

    for filename in os.listdir(DATASET_DIR):
        if not filename.lower().endswith(".jpg"):
            continue

        path = os.path.join(DATASET_DIR, filename)

        try:
            user_id = int(filename.split(".")[0])
        except Exception:
            print(f"Skipping invalid file: {filename}")
            continue

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        face_samples.append(img)
        ids.append(user_id)

    if not face_samples:
        print("No training data found.")
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8,
    )

    recognizer.train(face_samples, np.array(ids))
    recognizer.write(RECOGNIZER_PATH)

    print(f"Training complete: {len(set(ids))} users, {len(face_samples)} samples.")
    return True


# -------------------------
# ROUTES
# -------------------------
@register_bp.route("/", methods=["GET", "POST"])
def register_page():
    """Handles patient registration form."""
    if request.method == "POST":
        try:
            user_id = int(request.form["id"])
            name = request.form["name"]
            age = int(request.form["age"])
            gender = request.form["gender"]
            email = request.form["email"]
            password = request.form["password"]
            problem = request.form["problem"]
            appointment = request.form["appointment"]
            password_hash = generate_password_hash(password)
        except ValueError:
            flash("Invalid input. ID and Age must be numbers.", "danger")
            return render_template("register.html")

        try:
            insert_or_update(
                user_id,
                name,
                age,
                gender,
                email,
                password_hash,
                problem,
                appointment,
            )
        except sqlite3.IntegrityError:
            flash("Email already exists. Please use a different one.", "warning")
            return render_template("register.html")

        send_appointment_email(name, email, appointment, problem)
        flash("Registration successful! Please capture your face.", "success")

        return render_template("capture_face.html", user_id=user_id, name=name)

    return render_template("register.html")


@register_bp.route("/save_face", methods=["POST"])
def save_face():
    """Saves face samples as dataset/{user_id}.{sample_num}.jpg."""
    data = request.get_json()

    if not data:
        return jsonify({"status": "error", "message": "Invalid JSON data"})

    img_data = data.get("image")
    user_id = data.get("user_id")
    sample_num = data.get("sample_num")

    if not img_data or not user_id or sample_num is None:
        return jsonify({"status": "error", "message": "Invalid data"})

    user_id = int(user_id)
    sample_num = int(sample_num)

    img = decode_image(img_data)

    if img is None:
        return jsonify({"status": "error", "message": "Image decode failed"})

    face = extract_face(img)

    if face is None:
        return jsonify({"status": "noface", "message": "No face detected"})

    filename = os.path.join(DATASET_DIR, f"{user_id}.{sample_num:03d}.jpg")
    cv2.imwrite(filename, face)

    print(f"Saved: {filename}")

    if sample_num >= MAX_SAMPLES_PER_USER:
        print(f"Auto-training model after user {user_id} capture...")
        train_recognizer()

    return jsonify({"status": "success", "filename": filename})


@register_bp.route("/finish_capture", methods=["POST"])
def finish_capture():
    """Manual retraining endpoint."""
    try:
        trained = train_recognizer()

        if trained:
            return jsonify({"status": "trained"})

        return jsonify({"status": "error", "message": "No samples found to train."})

    except Exception as e:
        print("Training error:", e)
        return jsonify({"status": "error", "message": str(e)})
