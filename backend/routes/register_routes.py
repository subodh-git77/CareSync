# # # from flask import Blueprint, render_template, request, redirect, jsonify, flash
# # # from email_utils import send_appointment_email
# # # from werkzeug.security import generate_password_hash
# # # import os
# # # import sqlite3
# # # import cv2
# # # import base64
# # # import numpy as np

# # # # === Blueprint setup ===
# # # register_bp = Blueprint('register', __name__, template_folder='../templates')

# # # # === Dataset Path ===
# # # dataset_path = "dataset"
# # # os.makedirs(dataset_path, exist_ok=True)


# # # # === Helper: Connect DB ===
# # # def get_db_connection():
# # #     return sqlite3.connect("database.db")


# # # # === Helper: Insert or Update User ===
# # # def insert_or_update(id, name, age, gender, email, password_hash, problem, appointment):
# # #     conn = get_db_connection()
# # #     cursor = conn.execute("SELECT id FROM users WHERE id=? OR email=?", (id, email))
# # #     existing = cursor.fetchone()

# # #     if existing:
# # #         conn.execute("""
# # #             UPDATE users
# # #             SET name=?, age=?, gender=?, email=?, password_hash=?, problem=?, appointment_date=?
# # #             WHERE id=?
# # #         """, (name, age, gender, email, password_hash, problem, appointment, id))
# # #         print(f"🔄 Updated record for ID {id}")
# # #     else:
# # #         conn.execute("""
# # #             INSERT INTO users (id, name, age, gender, email, password_hash, problem, appointment_date)
# # #             VALUES (?, ?, ?, ?, ?, ?, ?, ?)
# # #         """, (id, name, age, gender, email, password_hash, problem, appointment))
# # #         print(f"🆕 Inserted new record for ID {id}")

# # #     conn.commit()
# # #     conn.close()


# # # # === Route: Register Page ===
# # # @register_bp.route('/', methods=['GET', 'POST'])
# # # def register_page():
# # #     if request.method == 'POST':
# # #         try:
# # #             id = int(request.form['id'])
# # #             name = request.form['name']
# # #             age = int(request.form['age'])
# # #             gender = request.form['gender']
# # #             email = request.form['email']
# # #             password = request.form['password']
# # #             problem = request.form['problem']
# # #             appointment = request.form['appointment']

# # #             password_hash = generate_password_hash(password)
# # #         except ValueError:
# # #             flash("❌ Invalid input. ID and Age must be numbers.", "danger")
# # #             return render_template('register.html')

# # #         try:
# # #             insert_or_update(id, name, age, gender, email, password_hash, problem, appointment)
# # #         except sqlite3.IntegrityError:
# # #             flash("⚠️ Email already exists. Please use a different one.", "warning")
# # #             return render_template('register.html')

# # #         send_appointment_email(name, email, appointment, problem)

# # #         # After form submission → Go to capture page
# # #         return render_template("capture_face.html", user_id=id, name=name)

# # #     return render_template('register.html')


# # # # === API Endpoint: Receive Face Frames (from frontend webcam) ===
# # # @register_bp.route('/save_face', methods=['POST'])
# # # def save_face():
# # #     data = request.get_json()
# # #     img_data = data.get('image')
# # #     user_id = data.get('user_id')
# # #     sample_num = data.get('sample_num')

# # #     if not img_data or not user_id:
# # #         return jsonify({"status": "error", "message": "Invalid data"})

# # #     # Decode base64 image
# # #     img_bytes = base64.b64decode(img_data.split(',')[1])
# # #     img_np = np.frombuffer(img_bytes, np.uint8)
# # #     img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

# # #     filename = os.path.join(dataset_path, f"{user_id}.{sample_num}.jpg")
# # #     cv2.imwrite(filename, img)
# # #     print(f"💾 Saved face image {filename}")

# # #     return jsonify({"status": "success", "filename": filename})


# # from flask import Blueprint, render_template, request, redirect, jsonify, flash
# # from email_utils import send_appointment_email
# # from werkzeug.security import generate_password_hash
# # import os
# # import sqlite3
# # import cv2
# # import base64
# # import numpy as np
# # from glob import glob

# # # === Blueprint setup ===
# # register_bp = Blueprint('register', __name__, template_folder='../templates')

# # # === Paths ===
# # DATASET_DIR = "dataset"
# # RECOGNIZER_DIR = "recognizer"
# # RECOGNIZER_PATH = os.path.join(RECOGNIZER_DIR, "trainingdata.yml")

# # os.makedirs(DATASET_DIR, exist_ok=True)
# # os.makedirs(RECOGNIZER_DIR, exist_ok=True)

# # # === Models ===
# # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # # How many samples do we want per user before training?
# # MAX_SAMPLES_PER_USER = 40


# # # -------------------------
# # # DB helpers
# # # -------------------------
# # def get_db_connection():
# #     # If your DB lives in backend/database.db, adjust this to an absolute path
# #     return sqlite3.connect("database.db")


# # def insert_or_update(id, name, age, gender, email, password_hash, problem, appointment):
# #     conn = get_db_connection()
# #     cursor = conn.execute("SELECT id FROM users WHERE id=? OR email=?", (id, email))
# #     existing = cursor.fetchone()

# #     if existing:
# #         conn.execute("""
# #             UPDATE users
# #             SET name=?, age=?, gender=?, email=?, password_hash=?, problem=?, appointment_date=?
# #             WHERE id=?
# #         """, (name, age, gender, email, password_hash, problem, appointment, id))
# #         print(f"🔄 Updated record for ID {id}")
# #     else:
# #         conn.execute("""
# #             INSERT INTO users (id, name, age, gender, email, password_hash, problem, appointment_date)
# #             VALUES (?, ?, ?, ?, ?, ?, ?, ?)
# #         """, (id, name, age, gender, email, password_hash, problem, appointment))
# #         print(f"🆕 Inserted new record for ID {id}")

# #     conn.commit()
# #     conn.close()


# # # -------------------------
# # # Image / training helpers
# # # -------------------------
# # def _decode_base64_image(img_data: str) -> np.ndarray | None:
# #     """Decode dataURL -> BGR image"""
# #     try:
# #         img_bytes = base64.b64decode(img_data.split(',')[1])
# #         img_np = np.frombuffer(img_bytes, np.uint8)
# #         img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
# #         return img
# #     except Exception as e:
# #         print("❌ Image decode error:", e)
# #         return None


# # def _extract_best_face_tile(bgr_img: np.ndarray, target_size=(200, 200)) -> np.ndarray | None:
# #     """
# #     Detects faces, picks the largest one, returns a cleaned grayscale face tile.
# #     Steps: grayscale -> equalize -> detect -> crop -> resize -> CLAHE normalize.
# #     """
# #     if bgr_img is None:
# #         return None

# #     gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
# #     gray = cv2.equalizeHist(gray)

# #     faces = face_cascade.detectMultiScale(
# #         gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
# #     if len(faces) == 0:
# #         return None

# #     # pick the largest face (most area)
# #     x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
# #     tile = gray[y:y + h, x:x + w]

# #     # normalize tile
# #     tile = cv2.resize(tile, target_size, interpolation=cv2.INTER_CUBIC)
# #     tile = cv2.GaussianBlur(tile, (3, 3), 0)
# #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# #     tile = clahe.apply(tile)

# #     return tile


# # def _count_user_samples(user_id: int) -> int:
# #     return len(glob(os.path.join(DATASET_DIR, f"{user_id}.*.jpg")))


# # def train_recognizer() -> bool:
# #     """Train LBPH recognizer from all tiles in dataset/ → recognizer/trainingdata.yml"""
# #     face_samples = []
# #     ids = []

# #     for filename in os.listdir(DATASET_DIR):
# #         if not filename.lower().endswith(".jpg"):
# #             continue
# #         path = os.path.join(DATASET_DIR, filename)
# #         try:
# #             user_id = int(filename.split(".")[0])  # 12.5.jpg -> 12
# #         except Exception:
# #             continue

# #         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# #         if img is None:
# #             continue
# #         face_samples.append(img)
# #         ids.append(user_id)

# #     if len(face_samples) == 0:
# #         print("⚠️ No face tiles found. Training skipped.")
# #         return False

# #     recognizer = cv2.face.LBPHFaceRecognizer_create(
# #         radius=1, neighbors=8, grid_x=8, grid_y=8)
# #     recognizer.train(face_samples, np.array(ids))
# #     recognizer.write(RECOGNIZER_PATH)
# #     print(f"✅ Training complete. Users in model: {len(set(ids))}, samples: {len(face_samples)}")
# #     return True


# # # -------------------------
# # # Routes
# # # -------------------------
# # @register_bp.route('/', methods=['GET', 'POST'])
# # def register_page():
# #     if request.method == 'POST':
# #         try:
# #             id = int(request.form['id'])
# #             name = request.form['name']
# #             age = int(request.form['age'])
# #             gender = request.form['gender']
# #             email = request.form['email']
# #             password = request.form['password']
# #             problem = request.form['problem']
# #             appointment = request.form['appointment']
# #             password_hash = generate_password_hash(password)
# #         except ValueError:
# #             flash("❌ Invalid input. ID and Age must be numbers.", "danger")
# #             return render_template('register.html')

# #         try:
# #             insert_or_update(id, name, age, gender, email, password_hash, problem, appointment)
# #         except sqlite3.IntegrityError:
# #             flash("⚠️ Email already exists. Please use a different one.", "warning")
# #             return render_template('register.html')

# #         # Appointment email
# #         send_appointment_email(name, email, appointment, problem)

# #         # Move to client-side capture page
# #         return render_template("capture_face.html", user_id=id, name=name)

# #     return render_template('register.html')


# # @register_bp.route('/save_face', methods=['POST'])
# # def save_face():
# #     """
# #     Receives base64 frame → detects/crops/normalizes the largest face →
# #     saves as dataset/{user_id}.{sample_num}.jpg (grayscale tile).
# #     Returns how many samples saved so far for this user.
# #     """
# #     data = request.get_json()
# #     img_data = data.get('image')
# #     user_id = data.get('user_id')
# #     sample_num = data.get('sample_num')

# #     if not img_data or not user_id or sample_num is None:
# #         return jsonify({"status": "error", "message": "Invalid data"})

# #     user_id = int(user_id)
# #     bgr = _decode_base64_image(img_data)
# #     tile = _extract_best_face_tile(bgr)

# #     if tile is None:
# #         # No face found in this frame, tell the client to keep trying
# #         return jsonify({"status": "noface", "saved": _count_user_samples(user_id)})

# #     filename = os.path.join(DATASET_DIR, f"{user_id}.{int(sample_num):03d}.jpg")
# #     cv2.imwrite(filename, tile)
# #     saved = _count_user_samples(user_id)
# #     print(f"💾 Saved face tile {filename}  (total for user {user_id}: {saved})")

# #     return jsonify({
# #         "status": "success",
# #         "filename": filename,
# #         "saved": saved,
# #         "target": MAX_SAMPLES_PER_USER
# #     })


# # @register_bp.route('/finish_capture', methods=['POST'])
# # def finish_capture():
# #     """
# #     Called by the frontend after reaching MAX_SAMPLES_PER_USER.
# #     Triggers training and returns result.
# #     """
# #     try:
# #         trained = train_recognizer()
# #         if trained:
# #             return jsonify({"status": "trained"})
# #         else:
# #             return jsonify({"status": "error", "message": "No samples found to train."})
# #     except Exception as e:
# #         print("❌ Training error:", e)
# #         return jsonify({"status": "error", "message": "Training failed."})

# from flask import Blueprint, render_template, request, jsonify, flash
# from email_utils import send_appointment_email
# from werkzeug.security import generate_password_hash
# import os
# import sqlite3
# import cv2
# import base64
# import numpy as np
# # from database import DB_PATH

# # === Blueprint setup ===
# register_bp = Blueprint('register', __name__, template_folder='../templates')

# # === Paths ===
# DATASET_DIR = "dataset"
# RECOGNIZER_DIR = "recognizer"
# RECOGNIZER_PATH = os.path.join(RECOGNIZER_DIR, "trainingdata.yml")

# # Ensure directories exist
# os.makedirs(DATASET_DIR, exist_ok=True)
# os.makedirs(RECOGNIZER_DIR, exist_ok=True)

# # === Face detector ===
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # === Constants ===
# MAX_SAMPLES_PER_USER = 40


# # -------------------------
# # DATABASE HELPERS
# # -------------------------
# def get_db_connection():
#     """Connect to main database."""
#     return sqlite3.connect("database.db")


# def insert_or_update(id, name, age, gender, email, password_hash, problem, appointment):
#     """Insert or update user details."""
#     conn = get_db_connection()
#     cursor = conn.execute("SELECT id FROM users WHERE id=? OR email=?", (id, email))
#     existing = cursor.fetchone()

#     if existing:
#         conn.execute("""
#             UPDATE users
#             SET name=?, age=?, gender=?, email=?, password_hash=?, problem=?, appointment_date=?
#             WHERE id=?
#         """, (name, age, gender, email, password_hash, problem, appointment, id))
#         print(f"🔄 Updated record for ID {id}")
#     else:
#         conn.execute("""
#             INSERT INTO users (id, name, age, gender, email, password_hash, problem, appointment_date)
#             VALUES (?, ?, ?, ?, ?, ?, ?, ?)
#         """, (id, name, age, gender, email, password_hash, problem, appointment))
#         print(f"🆕 Inserted new record for ID {id}")

#     conn.commit()
#     conn.close()


# # -------------------------
# # IMAGE HELPERS
# # -------------------------
# def decode_image(img_data: str):
#     """Decode base64-encoded image into OpenCV BGR image."""
#     try:
#         img_bytes = base64.b64decode(img_data.split(',')[1])
#         img_np = np.frombuffer(img_bytes, np.uint8)
#         return cv2.imdecode(img_np, cv2.IMREAD_COLOR)
#     except Exception as e:
#         print("❌ Failed to decode image:", e)
#         return None


# def extract_face(image, size=(200, 200)):
#     """Detect largest face and return preprocessed grayscale tile."""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)

#     faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(90, 90))
#     if len(faces) == 0:
#         return None

#     # choose the largest detected face
#     x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
#     face = gray[y:y + h, x:x + w]
#     face = cv2.resize(face, size, interpolation=cv2.INTER_CUBIC)

#     # normalize face using CLAHE (adaptive contrast)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     face = clahe.apply(face)

#     return face


# def train_recognizer():
#     """Train LBPH recognizer from flat files dataset/<user_id>.<sample>.jpg"""
#     face_samples, ids = [], []

#     for filename in os.listdir(DATASET_DIR):
#         if not filename.lower().endswith(".jpg"):
#             continue

#         path = os.path.join(DATASET_DIR, filename)
#         try:
#             user_id = int(filename.split(".")[0])  # filename format: 5.001.jpg
#         except Exception:
#             print(f"⚠️ Skipping invalid file: {filename}")
#             continue

#         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             continue

#         face_samples.append(img)
#         ids.append(user_id)

#     if not face_samples:
#         print("⚠️ No training data found.")
#         return False

#     recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
#     recognizer.train(face_samples, np.array(ids))
#     recognizer.write(RECOGNIZER_PATH)
#     print(f"✅ Training complete: {len(set(ids))} users, {len(face_samples)} samples.")
#     return True


# # -------------------------
# # ROUTES
# # -------------------------
# @register_bp.route('/', methods=['GET', 'POST'])
# def register_page():
#     """Handles patient registration form."""
#     if request.method == 'POST':
#         try:
#             id = int(request.form['id'])
#             name = request.form['name']
#             age = int(request.form['age'])
#             gender = request.form['gender']
#             email = request.form['email']
#             password = request.form['password']
#             problem = request.form['problem']
#             appointment = request.form['appointment']
#             password_hash = generate_password_hash(password)
#         except ValueError:
#             flash("❌ Invalid input. ID and Age must be numbers.", "danger")
#             return render_template('register.html')

#         try:
#             insert_or_update(id, name, age, gender, email, password_hash, problem, appointment)
#         except sqlite3.IntegrityError:
#             flash("⚠️ Email already exists. Please use a different one.", "warning")
#             return render_template('register.html')

#         send_appointment_email(name, email, appointment, problem)
#         flash("✅ Registration successful! Please capture your face.", "success")

#         return render_template("capture_face.html", user_id=id, name=name)

#     return render_template('register.html')


# @register_bp.route('/save_face', methods=['POST'])
# def save_face():
#     """Saves face samples in flat dataset format (dataset/{user_id}.{sample_num}.jpg)."""
#     data = request.get_json()
#     img_data = data.get('image')
#     user_id = data.get('user_id')
#     sample_num = data.get('sample_num')

#     if not img_data or not user_id or sample_num is None:
#         return jsonify({"status": "error", "message": "Invalid data"})

#     user_id = int(user_id)
#     sample_num = int(sample_num)

#     img = decode_image(img_data)
#     if img is None:
#         return jsonify({"status": "error", "message": "Image decode failed"})

#     face = extract_face(img)
#     if face is None:
#         return jsonify({"status": "noface", "message": "No face detected"})

#     filename = os.path.join(DATASET_DIR, f"{user_id}.{sample_num:03d}.jpg")
#     cv2.imwrite(filename, face)
#     print(f"💾 Saved: {filename}")

#     # Auto-train after final sample
#     if sample_num >= MAX_SAMPLES_PER_USER:
#         print(f"🚀 Auto-training model after user {user_id} capture...")
#         train_recognizer()

#     return jsonify({"status": "success", "filename": filename})


# @register_bp.route('/finish_capture', methods=['POST'])
# def finish_capture():
#     """Manual retraining endpoint (optional)."""
#     try:
#         trained = train_recognizer()
#         if trained:
#             return jsonify({"status": "trained"})
#         else:
#             return jsonify({"status": "error", "message": "No samples found to train."})
#     except Exception as e:
#         print("❌ Training error:", e)
#         return jsonify({"status": "error", "message": str(e)})







from flask import Blueprint, render_template, request, jsonify, flash
from email_utils import send_appointment_email
from werkzeug.security import generate_password_hash
import os
import sqlite3
import cv2
import base64
import numpy as np

# === Blueprint setup ===
register_bp = Blueprint('register', __name__, template_folder='../templates')

# === Paths ===
DATASET_DIR = "dataset"
RECOGNIZER_DIR = "recognizer"
RECOGNIZER_PATH = os.path.join(RECOGNIZER_DIR, "trainingdata.yml")

# ✅ Updated absolute database path
DB_PATH = r"C:\Users\subod\OneDrive\Desktop\careSync\Face Detection\backend\database.db"

# Ensure directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(RECOGNIZER_DIR, exist_ok=True)

# === Face detector ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === Constants ===
MAX_SAMPLES_PER_USER = 40


# -------------------------
# DATABASE HELPERS
# -------------------------
def get_db_connection():
    """Connect to main database."""
    return sqlite3.connect(DB_PATH)


def insert_or_update(id, name, age, gender, email, password_hash, problem, appointment):
    """Insert or update user details."""
    conn = get_db_connection()
    cursor = conn.execute("SELECT id FROM users WHERE id=? OR email=?", (id, email))
    existing = cursor.fetchone()

    if existing:
        conn.execute("""
            UPDATE users
            SET name=?, age=?, gender=?, email=?, password_hash=?, problem=?, appointment_date=?
            WHERE id=?
        """, (name, age, gender, email, password_hash, problem, appointment, id))
        print(f"🔄 Updated record for ID {id}")
    else:
        conn.execute("""
            INSERT INTO users (id, name, age, gender, email, password_hash, problem, appointment_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (id, name, age, gender, email, password_hash, problem, appointment))
        print(f"🆕 Inserted new record for ID {id}")

    conn.commit()
    conn.close()


# -------------------------
# IMAGE HELPERS
# -------------------------
def decode_image(img_data: str):
    """Decode base64-encoded image into OpenCV BGR image."""
    try:
        img_bytes = base64.b64decode(img_data.split(',')[1])
        img_np = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    except Exception as e:
        print("❌ Failed to decode image:", e)
        return None


def extract_face(image, size=(200, 200)):
    """Detect largest face and return preprocessed grayscale tile."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(90, 90))
    if len(faces) == 0:
        return None

    # choose the largest detected face
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    face = gray[y:y + h, x:x + w]
    face = cv2.resize(face, size, interpolation=cv2.INTER_CUBIC)

    # normalize face using CLAHE (adaptive contrast)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face = clahe.apply(face)

    return face


def train_recognizer():
    """Train LBPH recognizer from flat files dataset/<user_id>.<sample>.jpg"""
    face_samples, ids = [], []

    for filename in os.listdir(DATASET_DIR):
        if not filename.lower().endswith(".jpg"):
            continue

        path = os.path.join(DATASET_DIR, filename)
        try:
            user_id = int(filename.split(".")[0])  # filename format: 5.001.jpg
        except Exception:
            print(f"⚠️ Skipping invalid file: {filename}")
            continue

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        face_samples.append(img)
        ids.append(user_id)

    if not face_samples:
        print("⚠️ No training data found.")
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(face_samples, np.array(ids))
    recognizer.write(RECOGNIZER_PATH)
    print(f"✅ Training complete: {len(set(ids))} users, {len(face_samples)} samples.")
    return True


# -------------------------
# ROUTES
# -------------------------
@register_bp.route('/', methods=['GET', 'POST'])
def register_page():
    """Handles patient registration form."""
    if request.method == 'POST':
        try:
            id = int(request.form['id'])
            name = request.form['name']
            age = int(request.form['age'])
            gender = request.form['gender']
            email = request.form['email']
            password = request.form['password']
            problem = request.form['problem']
            appointment = request.form['appointment']
            password_hash = generate_password_hash(password)
        except ValueError:
            flash("❌ Invalid input. ID and Age must be numbers.", "danger")
            return render_template('register.html')

        try:
            insert_or_update(id, name, age, gender, email, password_hash, problem, appointment)
        except sqlite3.IntegrityError:
            flash("⚠️ Email already exists. Please use a different one.", "warning")
            return render_template('register.html')

        send_appointment_email(name, email, appointment, problem)
        flash("✅ Registration successful! Please capture your face.", "success")

        return render_template("capture_face.html", user_id=id, name=name)

    return render_template('register.html')


@register_bp.route('/save_face', methods=['POST'])
def save_face():
    """Saves face samples in flat dataset format (dataset/{user_id}.{sample_num}.jpg)."""
    data = request.get_json()
    img_data = data.get('image')
    user_id = data.get('user_id')
    sample_num = data.get('sample_num')

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
    print(f"💾 Saved: {filename}")

    # Auto-train after final sample
    if sample_num >= MAX_SAMPLES_PER_USER:
        print(f"🚀 Auto-training model after user {user_id} capture...")
        train_recognizer()

    return jsonify({"status": "success", "filename": filename})


@register_bp.route('/finish_capture', methods=['POST'])
def finish_capture():
    """Manual retraining endpoint (optional)."""
    try:
        trained = train_recognizer()
        if trained:
            return jsonify({"status": "trained"})
        else:
            return jsonify({"status": "error", "message": "No samples found to train."})
    except Exception as e:
        print("❌ Training error:", e)
        return jsonify({"status": "error", "message": str(e)})
