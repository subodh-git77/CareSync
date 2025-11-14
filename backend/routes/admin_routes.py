# from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
# import sqlite3
# import os
# import cv2
# import numpy as np
# import base64
# from werkzeug.security import check_password_hash
# from datetime import datetime

# # === Blueprint setup ===
# admin_bp = Blueprint('admin', __name__, template_folder='../templates')

# # === Database path ===
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DB_PATH = os.path.join(BASE_DIR, "../database.db")

# # === Face Recognition Model Path ===
# recognizer_path = os.path.join(os.path.dirname(BASE_DIR), "recognizer", "trainingdata.yml")
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # === Helper Function: Database Connection ===
# def get_db_connection():
#     return sqlite3.connect(DB_PATH)


# # ===============================
# # 🧑‍⚕️ ADMIN LOGIN
# # ===============================
# @admin_bp.route('/login', methods=['GET', 'POST'])
# def login_page():
#     if request.method == 'POST':
#         email = request.form.get('email')
#         password = request.form.get('password')

#         if not email or not password:
#             flash("⚠️ Please fill in all fields.", "warning")
#             return render_template('admin_login.html')

#         conn = get_db_connection()
#         admin = conn.execute(
#             "SELECT id, name, email, password_hash FROM admins WHERE email=?",
#             (email,)
#         ).fetchone()
#         conn.close()

#         if admin and check_password_hash(admin[3], password):
#             session['admin_id'] = admin[0]
#             session['admin_name'] = admin[1]
#             flash(f"✅ Welcome back, {admin[1]}!", "success")
#             return redirect(url_for('admin.dashboard'))
#         else:
#             flash("❌ Invalid credentials. Please try again.", "danger")

#     return render_template('admin_login.html')


# # ===============================
# # 📊 ADMIN DASHBOARD (With Approve Button)
# # ===============================
# @admin_bp.route('/dashboard')
# def dashboard():
#     if 'admin_id' not in session:
#         flash("⚠️ Please login first.", "warning")
#         return redirect(url_for('admin.login_page'))

#     conn = get_db_connection()
#     users = conn.execute("""
#         SELECT id, name, email, problem, appointment_date, COALESCE(approved, 0)
#         FROM users ORDER BY appointment_date
#     """).fetchall()

#     today = datetime.now().strftime("%Y-%m-%d")
#     today_appointments = conn.execute(
#         "SELECT COUNT(*) FROM users WHERE appointment_date=?", (today,)
#     ).fetchone()[0]
#     total_patients = len(users)
#     approved_count = sum(1 for u in users if u[5] == 1)
#     conn.close()

#     stats = {
#         "total_patients": total_patients,
#         "today_appointments": today_appointments,
#         "total_appointments": total_patients,
#         "pending_visits": total_patients - approved_count,
#         "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     }

#     return render_template(
#         'admin_dashboard.html',
#         users=users,
#         admin_name=session.get('admin_name'),
#         stats=stats
#     )


# # ===============================
# # 🔍 VERIFY FACE FOR APPROVAL PAGE
# # ===============================
# @admin_bp.route('/verify_user/<int:user_id>')
# def verify_user(user_id):
#     if 'admin_id' not in session:
#         flash("⚠️ Please login first.", "warning")
#         return redirect(url_for('admin.login_page'))

#     conn = get_db_connection()
#     user = conn.execute("SELECT id, name FROM users WHERE id=?", (user_id,)).fetchone()
#     conn.close()

#     if not user:
#         flash("❌ User not found.", "danger")
#         return redirect(url_for('admin.dashboard'))

#     return render_template('admin_verify_user.html', user_id=user[0], name=user[1])


# # # ===============================
# # # 🧠 VERIFY USER FACE (AJAX Endpoint)
# # # ===============================
# # @admin_bp.route('/verify_user_frame', methods=['POST'])
# # def verify_user_frame():
# #     if 'admin_id' not in session:
# #         return jsonify({"status": "error", "message": "Unauthorized"}), 401

# #     if not os.path.exists(recognizer_path):
# #         return jsonify({"status": "error", "message": "⚠️ Model not trained yet."})

# #     data = request.get_json()
# #     img_data = data.get('image')
# #     user_id = int(data.get('user_id'))

# #     # Decode base64 image
# #     try:
# #         img_bytes = base64.b64decode(img_data.split(',')[1])
# #         img_np = np.frombuffer(img_bytes, np.uint8)
# #         img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
# #     except Exception as e:
# #         return jsonify({"status": "error", "message": f"Image decode failed: {e}"})

# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #     gray = cv2.equalizeHist(gray)

# #     faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
# #     if len(faces) == 0:
# #         return jsonify({"status": "no_face"})

# #     recognizer = cv2.face.LBPHFaceRecognizer_create()
# #     recognizer.read(recognizer_path)

# #     verified = False
# #     for (x, y, w, h) in faces:
# #         face_roi = gray[y:y+h, x:x+w]
# #         if face_roi.size == 0:
# #             continue

# #         face_resized = cv2.resize(face_roi, (200, 200))
# #         pred_id, conf = recognizer.predict(face_resized)
# #         print(f"Prediction → ID: {pred_id}, Conf: {conf:.2f}")

# #         if pred_id == user_id and conf < 75:
# #             verified = True
# #             break

# #     if verified:
# #         conn = get_db_connection()
# #         conn.execute("UPDATE users SET approved=1 WHERE id=?", (user_id,))
# #         conn.commit()
# #         conn.close()
# #         print(f"✅ User {user_id} approved successfully.")
# #         flash("✅ Patient verified and approved successfully!", "success")
# #         return jsonify({"status": "approved"})
# #     else:
# #         print(f"❌ Verification failed for user {user_id}")
# #         flash("❌ Face verification failed. Please try again.", "danger")
# #         return jsonify({"status": "failed"})


# # ===============================
# # 🧠 VERIFY USER FACE (AJAX Endpoint)
# # ===============================
# @admin_bp.route('/verify_user_frame', methods=['POST'])
# def verify_user_frame():
#     if 'admin_id' not in session:
#         return jsonify({"status": "error", "message": "Unauthorized"}), 401

#     if not os.path.exists(recognizer_path):
#         return jsonify({"status": "error", "message": "⚠️ Model not trained yet."})

#     data = request.get_json()
#     img_data = data.get('image')
#     user_id = int(data.get('user_id'))

#     # Decode base64 image
#     try:
#         img_bytes = base64.b64decode(img_data.split(',')[1])
#         img_np = np.frombuffer(img_bytes, np.uint8)
#         img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
#     except Exception as e:
#         return jsonify({"status": "error", "message": f"Image decode failed: {e}"})

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)

#     # Detect faces
#     faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(90, 90))
#     if len(faces) == 0:
#         print("⚠️ No face detected.")
#         return jsonify({"status": "no_face"})

#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.read(recognizer_path)

#     verified = False
#     confidence_score = 999  # large default

#     for (x, y, w, h) in faces:
#         face_roi = gray[y:y+h, x:x+w]
#         if face_roi.size == 0:
#             continue

#         face_resized = cv2.resize(face_roi, (200, 200))
#         pred_id, conf = recognizer.predict(face_resized)
#         print(f"Prediction → ID: {pred_id}, Conf: {conf:.2f}")
#         confidence_score = conf

#         # ✅ Increased accuracy threshold: 65 for strong match
#         if pred_id == user_id and conf < 60:
#             verified = True
#             break

#     conn = get_db_connection()

#     if verified:
#         conn.execute("UPDATE users SET approved=1 WHERE id=?", (user_id,))
#         conn.commit()
#         conn.close()
#         print(f"✅ Face Verified & Approved for user {user_id}")
#         flash("✅ Patient verified successfully!", "success")
#         return jsonify({"status": "approved"})
#     else:
#         # ❌ Wrong face — DO NOT APPROVE
#         conn.execute("UPDATE users SET approved=0 WHERE id=?", (user_id,))
#         conn.commit()
#         conn.close()
#         print(f"❌ Verification failed for user {user_id} (Conf={confidence_score:.2f})")
#         flash("❌ Verification failed! Wrong person detected. Returning to dashboard.", "danger")

#         # 🔁 Return dashboard redirect
#         return jsonify({"status": "failed", "redirect": url_for('admin.dashboard')})


# # ===============================
# # 🚪 LOGOUT
# # ===============================
# @admin_bp.route('/logout')
# def logout():
#     session.pop('admin_id', None)
#     session.pop('admin_name', None)
#     flash("👋 You’ve been logged out successfully.", "info")
#     return redirect(url_for('admin.login_page'))


# from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
# import sqlite3
# import os
# import cv2
# import numpy as np
# import base64
# from werkzeug.security import check_password_hash
# from datetime import datetime

# # === Blueprint setup ===
# admin_bp = Blueprint('admin', __name__, template_folder='../templates')

# # === Paths ===
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))                 # .../backend/routes
# ROOT_DIR = os.path.dirname(BASE_DIR)                                  # .../backend
# DB_PATH = os.path.join(ROOT_DIR, "database.db")                       # backend/database.db
# RECOGNIZER_PATH = os.path.join(ROOT_DIR, "recognizer", "trainingdata.yml")

# # === Models ===
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # === Tuning ===
# CONF_THRESHOLD = 58             # lower = stricter, typical 55–70 for LBPH. We choose stricter.
# MIN_FACE = (100, 100)           # larger minSize reduces false detections
# SCALE_FACTOR = 1.15
# MIN_NEIGHBORS = 6
# FACE_SIZE = (200, 200)          # must match training tile size

# # === DB helper ===
# def get_db_connection():
#     return sqlite3.connect(DB_PATH)

# # ===============================
# # 🧑‍⚕️ ADMIN LOGIN
# # ===============================
# @admin_bp.route('/login', methods=['GET', 'POST'])
# def login_page():
#     if request.method == 'POST':
#         email = request.form.get('email', '').strip()
#         password = request.form.get('password', '')

#         if not email or not password:
#             flash("⚠️ Please fill in all fields.", "warning")
#             return render_template('admin_login.html')

#         conn = get_db_connection()
#         admin = conn.execute(
#             "SELECT id, name, email, password_hash FROM admins WHERE email=?",
#             (email,)
#         ).fetchone()
#         conn.close()

#         if admin and admin[3] and check_password_hash(admin[3], password):
#             session['admin_id'] = admin[0]
#             session['admin_name'] = admin[1]
#             flash(f"✅ Welcome back, {admin[1]}!", "success")
#             return redirect(url_for('admin.dashboard'))
#         else:
#             flash("❌ Invalid credentials. Please try again.", "danger")

#     return render_template('admin_login.html')

# # ===============================
# # 📊 ADMIN DASHBOARD (Approve column)
# # ===============================
# @admin_bp.route('/dashboard')
# def dashboard():
#     if 'admin_id' not in session:
#         flash("⚠️ Please login first.", "warning")
#         return redirect(url_for('admin.login_page'))

#     conn = get_db_connection()
#     users = conn.execute("""
#         SELECT id, name, email, problem, appointment_date, COALESCE(approved, 0)
#         FROM users
#         ORDER BY appointment_date
#     """).fetchall()

#     today = datetime.now().strftime("%Y-%m-%d")
#     today_appointments = conn.execute(
#         "SELECT COUNT(*) FROM users WHERE appointment_date = ?",
#         (today,)
#     ).fetchone()[0]
#     total_patients = len(users)
#     approved_count = sum(1 for u in users if int(u[5]) == 1)
#     conn.close()

#     stats = {
#         "total_patients": total_patients,
#         "today_appointments": today_appointments,
#         "total_appointments": total_patients,
#         "pending_visits": max(total_patients - approved_count, 0),
#         "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     }

#     return render_template(
#         'admin_dashboard.html',
#         users=users,
#         admin_name=session.get('admin_name'),
#         stats=stats
#     )

# # ===============================
# # 🔍 VERIFY PAGE (admin triggers webcam)
# # ===============================
# @admin_bp.route('/verify_user/<int:user_id>')
# def verify_user(user_id):
#     if 'admin_id' not in session:
#         flash("⚠️ Please login first.", "warning")
#         return redirect(url_for('admin.login_page'))

#     conn = get_db_connection()
#     user = conn.execute("SELECT id, name FROM users WHERE id=?", (user_id,)).fetchone()
#     conn.close()

#     if not user:
#         flash("❌ User not found.", "danger")
#         return redirect(url_for('admin.dashboard'))

#     return render_template('admin_verify_user.html', user_id=user[0], name=user[1])

# # ===============================
# # 🧠 VERIFY (AJAX) — strict + fast, safe fallback
# # ===============================
# @admin_bp.route('/verify_user_frame', methods=['POST'])
# def verify_user_frame():
#     if 'admin_id' not in session:
#         return jsonify({"status": "error", "message": "Unauthorized"}), 401

#     if not os.path.exists(RECOGNIZER_PATH):
#         return jsonify({"status": "error", "message": "⚠️ Model not trained yet."})

#     data = request.get_json() or {}
#     img_data = data.get('image')
#     try:
#         user_id = int(data.get('user_id'))
#     except Exception:
#         return jsonify({"status": "error", "message": "Invalid user id"})

#     # --- decode incoming frame
#     try:
#         img_bytes = base64.b64decode(img_data.split(',')[1])
#         img_np = np.frombuffer(img_bytes, np.uint8)
#         frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
#         if frame is None:
#             raise ValueError("Empty frame")
#     except Exception as e:
#         return jsonify({"status": "error", "message": f"Image decode failed: {e}"})

#     # --- preprocess
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)

#     # --- detect faces
#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=SCALE_FACTOR,
#         minNeighbors=MIN_NEIGHBORS,
#         minSize=MIN_FACE
#     )
#     if len(faces) == 0:
#         # no UI change; let frontend keep trying
#         return jsonify({"status": "no_face"})

#     # --- load recognizer
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.read(RECOGNIZER_PATH)

#     # --- verify strictly
#     verified = False
#     best_conf = 999.0
#     best_pred = None

#     for (x, y, w, h) in faces:
#         roi = gray[y:y+h, x:x+w]
#         if roi.size == 0:
#             continue
#         face_resized = cv2.resize(roi, FACE_SIZE)
#         pred_id, conf = recognizer.predict(face_resized)
#         best_conf = min(best_conf, conf)
#         if pred_id == user_id and conf < CONF_THRESHOLD:
#             verified = True
#             best_pred = pred_id
#             break

#     conn = get_db_connection()
#     if verified:
#         conn.execute("UPDATE users SET approved=1 WHERE id=?", (user_id,))
#         conn.commit()
#         conn.close()
#         # success → frontend will mark green and return to dashboard
#         return jsonify({"status": "approved"})
#     else:
#         # ensure not approved on failure
#         conn.execute("UPDATE users SET approved=0 WHERE id=?", (user_id,))
#         conn.commit()
#         conn.close()
#         # include a redirect so frontend can bounce back to dashboard fast
#         return jsonify({
#             "status": "failed",
#             "conf": float(best_conf),
#             "pred": int(best_pred) if best_pred is not None else None,
#             "redirect": url_for('admin.dashboard')
#         })

# # ===============================
# # 🚪 LOGOUT
# # ===============================
# @admin_bp.route('/logout')
# def logout():
#     session.pop('admin_id', None)
#     session.pop('admin_name', None)
#     flash("👋 You’ve been logged out successfully.", "info")
#     return redirect(url_for('admin.login_page'))











from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
import sqlite3
import os
import cv2
import numpy as np
import base64
from werkzeug.security import check_password_hash
from datetime import datetime

# === Blueprint setup ===
admin_bp = Blueprint('admin', __name__, template_folder='../templates')

# === Absolute Paths ===
DB_PATH = r"C:\Users\subod\OneDrive\Desktop\careSync\Face Detection\backend\database.db"
RECOGNIZER_PATH = r"C:\Users\subod\OneDrive\Desktop\careSync\Face Detection\recognizer\trainingdata.yml"
DATASET_DIR = r"C:\Users\subod\OneDrive\Desktop\careSync\Face Detection\dataset"

# === Models ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === Tuning ===
CONF_THRESHOLD = 58
MIN_FACE = (100, 100)
SCALE_FACTOR = 1.15
MIN_NEIGHBORS = 6
FACE_SIZE = (200, 200)

# === DB helper ===
def get_db_connection():
    return sqlite3.connect(DB_PATH)

# ===============================
# 🧑‍⚕️ ADMIN LOGIN
# ===============================
@admin_bp.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        if not email or not password:
            flash("⚠️ Please fill in all fields.", "warning")
            return render_template('admin_login.html')

        conn = get_db_connection()
        admin = conn.execute(
            "SELECT id, name, email, password_hash FROM admins WHERE email=?",
            (email,)
        ).fetchone()
        conn.close()

        if admin and admin[3] and check_password_hash(admin[3], password):
            session['admin_id'] = admin[0]
            session['admin_name'] = admin[1]
            flash(f"✅ Welcome back, {admin[1]}!", "success")
            return redirect(url_for('admin.dashboard'))
        else:
            flash("❌ Invalid credentials. Please try again.", "danger")

    return render_template('admin_login.html')

# ===============================
# 📊 ADMIN DASHBOARD
# ===============================
@admin_bp.route('/dashboard')
def dashboard():
    if 'admin_id' not in session:
        flash("⚠️ Please login first.", "warning")
        return redirect(url_for('admin.login_page'))

    conn = get_db_connection()
    users = conn.execute("""
        SELECT id, name, email, problem, appointment_date, COALESCE(approved, 0)
        FROM users
        ORDER BY appointment_date
    """).fetchall()

    today = datetime.now().strftime("%Y-%m-%d")
    today_appointments = conn.execute(
        "SELECT COUNT(*) FROM users WHERE appointment_date = ?",
        (today,)
    ).fetchone()[0]
    total_patients = len(users)
    approved_count = sum(1 for u in users if int(u[5]) == 1)
    conn.close()

    stats = {
        "total_patients": total_patients,
        "today_appointments": today_appointments,
        "total_appointments": total_patients,
        "pending_visits": max(total_patients - approved_count, 0),
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return render_template(
        'admin_dashboard.html',
        users=users,
        admin_name=session.get('admin_name'),
        stats=stats
    )

# ===============================
# 🔍 VERIFY PAGE
# ===============================
@admin_bp.route('/verify_user/<int:user_id>')
def verify_user(user_id):
    if 'admin_id' not in session:
        flash("⚠️ Please login first.", "warning")
        return redirect(url_for('admin.login_page'))

    conn = get_db_connection()
    user = conn.execute("SELECT id, name FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()

    if not user:
        flash("❌ User not found.", "danger")
        return redirect(url_for('admin.dashboard'))

    return render_template('admin_verify_user.html', user_id=user[0], name=user[1])

# ===============================
# 🧠 VERIFY (AJAX)
# ===============================
@admin_bp.route('/verify_user_frame', methods=['POST'])
def verify_user_frame():
    if 'admin_id' not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    if not os.path.exists(RECOGNIZER_PATH):
        return jsonify({"status": "error", "message": "⚠️ Model not trained yet."})

    data = request.get_json() or {}
    img_data = data.get('image')
    try:
        user_id = int(data.get('user_id'))
    except Exception:
        return jsonify({"status": "error", "message": "Invalid user id"})

    try:
        img_bytes = base64.b64decode(img_data.split(',')[1])
        img_np = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Empty frame")
    except Exception as e:
        return jsonify({"status": "error", "message": f"Image decode failed: {e}"})

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=MIN_FACE
    )
    if len(faces) == 0:
        return jsonify({"status": "no_face"})

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(RECOGNIZER_PATH)

    verified = False
    best_conf = 999.0
    best_pred = None

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        face_resized = cv2.resize(roi, FACE_SIZE)
        pred_id, conf = recognizer.predict(face_resized)
        best_conf = min(best_conf, conf)
        if pred_id == user_id and conf < CONF_THRESHOLD:
            verified = True
            best_pred = pred_id
            break

    conn = get_db_connection()
    if verified:
        conn.execute("UPDATE users SET approved=1 WHERE id=?", (user_id,))
        conn.commit()
        conn.close()
        return jsonify({"status": "approved"})
    else:
        conn.execute("UPDATE users SET approved=0 WHERE id=?", (user_id,))
        conn.commit()
        conn.close()
        return jsonify({
            "status": "failed",
            "conf": float(best_conf),
            "pred": int(best_pred) if best_pred is not None else None,
            "redirect": url_for('admin.dashboard')
        })

# ===============================
# 🚪 LOGOUT
# ===============================
@admin_bp.route('/logout')
def logout():
    session.pop('admin_id', None)
    session.pop('admin_name', None)
    flash("👋 You’ve been logged out successfully.", "info")
    return redirect(url_for('admin.login_page'))