from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import check_password_hash
from datetime import datetime

import sqlite3
import os
import cv2
import numpy as np
import base64

# === Blueprint setup ===
admin_bp = Blueprint("admin", __name__, template_folder="../templates")

# === Base Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_PATH = os.path.join(BASE_DIR, "database.db")
RECOGNIZER_DIR = os.path.join(BASE_DIR, "recognizer")
RECOGNIZER_PATH = os.path.join(RECOGNIZER_DIR, "trainingdata.yml")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

os.makedirs(RECOGNIZER_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# === Models ===
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# === Tuning ===
CONF_THRESHOLD = 58
MIN_FACE = (100, 100)
SCALE_FACTOR = 1.15
MIN_NEIGHBORS = 6
FACE_SIZE = (200, 200)


# ===============================
# DATABASE HELPER
# ===============================
def get_db_connection():
    return sqlite3.connect(DB_PATH)


# ===============================
# ADMIN LOGIN
# ===============================
@admin_bp.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Please fill in all fields.", "warning")
            return render_template("admin_login.html")

        conn = get_db_connection()
        admin = conn.execute(
            "SELECT id, name, email, password_hash FROM admins WHERE email=?",
            (email,),
        ).fetchone()
        conn.close()

        if admin and admin[3] and check_password_hash(admin[3], password):
            session["admin_id"] = admin[0]
            session["admin_name"] = admin[1]
            flash(f"Welcome back, {admin[1]}!", "success")
            return redirect(url_for("admin.dashboard"))

        flash("Invalid credentials. Please try again.", "danger")

    return render_template("admin_login.html")


# ===============================
# ADMIN DASHBOARD
# ===============================
@admin_bp.route("/dashboard")
def dashboard():
    if "admin_id" not in session:
        flash("Please login first.", "warning")
        return redirect(url_for("admin.login_page"))

    conn = get_db_connection()

    users = conn.execute(
        """
        SELECT id, name, email, problem, appointment_date, COALESCE(approved, 0)
        FROM users
        ORDER BY appointment_date
        """
    ).fetchall()

    today = datetime.now().strftime("%Y-%m-%d")

    today_appointments = conn.execute(
        "SELECT COUNT(*) FROM users WHERE appointment_date = ?",
        (today,),
    ).fetchone()[0]

    total_patients = len(users)
    approved_count = sum(1 for user in users if int(user[5]) == 1)

    conn.close()

    stats = {
        "total_patients": total_patients,
        "today_appointments": today_appointments,
        "total_appointments": total_patients,
        "pending_visits": max(total_patients - approved_count, 0),
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return render_template(
        "admin_dashboard.html",
        users=users,
        admin_name=session.get("admin_name"),
        stats=stats,
    )


# ===============================
# VERIFY PAGE
# ===============================
@admin_bp.route("/verify_user/<int:user_id>")
def verify_user(user_id):
    if "admin_id" not in session:
        flash("Please login first.", "warning")
        return redirect(url_for("admin.login_page"))

    conn = get_db_connection()
    user = conn.execute(
        "SELECT id, name FROM users WHERE id=?",
        (user_id,),
    ).fetchone()
    conn.close()

    if not user:
        flash("User not found.", "danger")
        return redirect(url_for("admin.dashboard"))

    return render_template(
        "admin_verify_user.html",
        user_id=user[0],
        name=user[1],
    )


# ===============================
# VERIFY USER FACE AJAX
# ===============================
@admin_bp.route("/verify_user_frame", methods=["POST"])
def verify_user_frame():
    if "admin_id" not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    if not os.path.exists(RECOGNIZER_PATH):
        return jsonify({"status": "error", "message": "Model not trained yet."})

    data = request.get_json() or {}
    img_data = data.get("image")

    try:
        user_id = int(data.get("user_id"))
    except Exception:
        return jsonify({"status": "error", "message": "Invalid user id"})

    if not img_data:
        return jsonify({"status": "error", "message": "Image data missing"})

    try:
        img_bytes = base64.b64decode(img_data.split(",")[1])
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
        minSize=MIN_FACE,
    )

    if len(faces) == 0:
        return jsonify({"status": "no_face"})

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(RECOGNIZER_PATH)

    verified = False
    best_conf = 999.0
    best_pred = None

    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]

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

    conn.execute("UPDATE users SET approved=0 WHERE id=?", (user_id,))
    conn.commit()
    conn.close()

    return jsonify(
        {
            "status": "failed",
            "conf": float(best_conf),
            "pred": int(best_pred) if best_pred is not None else None,
            "redirect": url_for("admin.dashboard"),
        }
    )


# ===============================
# LOGOUT
# ===============================
@admin_bp.route("/logout")
def logout():
    session.pop("admin_id", None)
    session.pop("admin_name", None)

    flash("You have been logged out successfully.", "info")
    return redirect(url_for("admin.login_page"))
