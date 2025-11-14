# from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
# from werkzeug.security import check_password_hash, generate_password_hash
# import sqlite3, cv2, os, numpy as np, base64, random, dlib
# from datetime import datetime
# from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
# from email_utils import send_reset_email  # make sure this file exists and works

# # === Blueprint ===
# patient_bp = Blueprint('patient', __name__, template_folder='../templates')

# # === Paths ===
# RECOGNIZER_PATH = os.path.join("recognizer", "trainingdata.yml")
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # === Dlib Detector (Optional smoother detection) ===
# detector = dlib.get_frontal_face_detector()
# sp = None
# try:
#     sp_path = os.path.join("recognizer", "shape_predictor_5_face_landmarks.dat")
#     if os.path.exists(sp_path):
#         sp = dlib.shape_predictor(sp_path)
# except Exception:
#     sp = None

# # === Security Token Setup ===
# SECRET_KEY = "caresync_secret_key"
# serializer = URLSafeTimedSerializer(SECRET_KEY)

# # === Database Connection ===
# def get_db():
#     db_path = os.path.join(os.getcwd(), "database.db")
#     return sqlite3.connect(db_path)


# # ===============================
# # 🧑‍💻 PATIENT LOGIN
# # ===============================
# @patient_bp.route('/login', methods=['GET', 'POST'])
# def login_page():
#     if request.method == 'POST':
#         email = request.form['email'].strip().lower()
#         password = request.form['password']

#         conn = get_db()
#         user = conn.execute(
#             "SELECT id, name, email, password_hash FROM users WHERE email=?",
#             (email,)
#         ).fetchone()
#         conn.close()

#         if user and user[3] and check_password_hash(user[3], password):
#             session['user_id'] = user[0]
#             session['user_name'] = user[1]
#             flash(f"Welcome {user[1]}! Please verify your face.", "info")
#             return redirect(url_for('patient.face_verify_page'))
#         else:
#             flash("❌ Invalid email or password.", "danger")

#     return render_template('patient_login.html')


# # ===============================
# # 🧠 FACE VERIFICATION PAGE
# # ===============================
# @patient_bp.route('/verify', methods=['GET'])
# def face_verify_page():
#     if 'user_id' not in session:
#         flash("Please login first.", "warning")
#         return redirect(url_for('patient.login_page'))
#     return render_template('verify_face.html', user_id=session['user_id'], name=session['user_name'])


# # ===============================
# # ⚡ FACE VERIFICATION (AJAX)
# # ===============================
# # @patient_bp.route('/verify_frame', methods=['POST'])
# # def verify_frame():
# #     data = request.get_json()
# #     img_data = data.get('image')
# #     user_id = int(data.get('user_id'))

# #     # Check if model exists
# #     if not os.path.exists(RECOGNIZER_PATH):
# #         return jsonify({"status": "error", "message": "⚠️ Face model not trained yet."})

# #     # Decode base64 image from webcam
# #     try:
# #         img_bytes = base64.b64decode(img_data.split(',')[1])
# #         img_np = np.frombuffer(img_bytes, np.uint8)
# #         img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
# #     except Exception as e:
# #         print("❌ Image decode error:", e)
# #         return jsonify({"status": "error", "message": "Failed to read image data"})

# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #     gray = cv2.equalizeHist(gray)

# #     # Detect faces (use OpenCV Haar)
# #     faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(90, 90))
# #     if len(faces) == 0:
# #         print("⚠️ No face detected in frame.")
# #         return jsonify({"status": "no_face"})

# #     # Load recognizer model
# #     recognizer = cv2.face.LBPHFaceRecognizer_create()
# #     recognizer.read(RECOGNIZER_PATH)

# #     verified = False
# #     confidence_score = 999
# #     predicted_id = None

# #     for (x, y, w, h) in faces:
# #         face_roi = gray[y:y+h, x:x+w]
# #         if face_roi.size == 0:
# #             continue

# #         face_resized = cv2.resize(face_roi, (200, 200))
# #         id_pred, conf = recognizer.predict(face_resized)
# #         predicted_id = id_pred
# #         confidence_score = conf
# #         print(f"Prediction → ID: {id_pred}, Conf: {conf:.2f}")

# #         # Relaxed threshold
# #         if id_pred == user_id and conf < 85:
# #             verified = True
# #             print(f"✅ Verified match: ID={id_pred}, Confidence={conf:.2f}")
# #             break
# #         else:
# #             print(f"⚠️ Mismatch or low confidence → Pred: {id_pred}, Conf={conf:.2f}")

# #     # Decision based on verification
# #     if verified:
# #         session['verified'] = True
# #         print(f"✅ Face verified successfully for user {user_id}")
# #         return jsonify({"status": "verified"})
# #     else:
# #         # Handle borderline or failure case
# #         if predicted_id == user_id and confidence_score < 95:
# #             print("⚠️ Accepting borderline match due to repeated recognition")
# #             session['verified'] = True
# #             return jsonify({"status": "verified"})
# #         print(f"❌ Verification failed. Conf={confidence_score:.2f}")
# #         return jsonify({"status": "failed"})

# @patient_bp.route('/verify_frame', methods=['POST'])
# def verify_frame():
#     data = request.get_json()
#     img_data = data.get('image')
#     user_id = int(data.get('user_id'))

#     # Check if model exists
#     if not os.path.exists(RECOGNIZER_PATH):
#         return jsonify({"status": "error", "message": "⚠️ Face model not trained yet."})

#     # Decode base64 image
#     try:
#         img_bytes = base64.b64decode(img_data.split(',')[1])
#         img_np = np.frombuffer(img_bytes, np.uint8)
#         img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
#     except Exception as e:
#         print("❌ Image decode error:", e)
#         return jsonify({"status": "error", "message": "Failed to decode image"})

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)

#     # Detect faces
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(90, 90))
#     if len(faces) == 0:
#         return jsonify({"status": "no_face"})

#     # Load trained recognizer
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.read(RECOGNIZER_PATH)

#     confidences = []
#     predictions = []

#     for (x, y, w, h) in faces:
#         face_roi = gray[y:y+h, x:x+w]
#         if face_roi.size == 0:
#             continue

#         face_resized = cv2.resize(face_roi, (200, 200))
#         pred_id, conf = recognizer.predict(face_resized)
#         predictions.append(pred_id)
#         confidences.append(conf)

#     if not confidences:
#         return jsonify({"status": "no_face"})

#     avg_conf = sum(confidences) / len(confidences)
#     most_common_id = max(set(predictions), key=predictions.count)

#     print(f"🧩 Frame analysis → PredID={most_common_id}, AvgConf={avg_conf:.2f}, ExpectedID={user_id}")

#     # ✅ Decision logic
#     if most_common_id == user_id and avg_conf < 75:
#         session['verified'] = True
#         print(f"✅ Verified: ID={most_common_id}, Confidence={avg_conf:.2f}")
#         return jsonify({"status": "verified"})
#     elif most_common_id == user_id and avg_conf < 95:
#         # borderline match, accept for prototype
#         session['verified'] = True
#         print(f"⚠️ Borderline but accepted: ID={most_common_id}, Conf={avg_conf:.2f}")
#         return jsonify({"status": "verified"})
#     else:
#         print(f"❌ Failed verification → Pred={most_common_id}, Conf={avg_conf:.2f}")
#         return jsonify({"status": "failed"})

# # ===============================
# # 📋 PATIENT DETAILS PAGE
# # ===============================
# @patient_bp.route('/details')
# def details():
#     if 'user_id' not in session:
#         flash("Please login first.", "warning")
#         return redirect(url_for('patient.login_page'))

#     if not session.get('verified'):
#         flash("Please verify your face first.", "warning")
#         return redirect(url_for('patient.face_verify_page'))

#     conn = get_db()
#     user = conn.execute("""
#         SELECT name, email, problem, appointment_date
#         FROM users WHERE id=?
#     """, (session['user_id'],)).fetchone()
#     conn.close()

#     if not user:
#         flash("User not found!", "danger")
#         return redirect(url_for('patient.login_page'))

#     doctors = [
#         "Dr. John Smith", "Dr. Jennifer Williams", "Dr. Michael Brown",
#         "Dr. Sarah Davis", "Dr. Robert Miller"
#     ]
#     time_slots = ["09:00 AM", "10:30 AM", "01:00 PM", "03:30 PM", "05:00 PM"]

#     data = {
#         "name": user[0],
#         "email": user[1],
#         "problem": user[2],
#         "doctor": random.choice(doctors),
#         "date": user[3] or datetime.now().strftime("%Y-%m-%d"),
#         "time": random.choice(time_slots),
#         "status": "booked"
#     }

#     # Log visit
#     try:
#         conn = get_db()
#         conn.execute("""
#             INSERT INTO visits (user_id, checkin_time, recognized)
#             VALUES (?, datetime('now'), 1)
#         """, (session['user_id'],))
#         conn.commit()
#         conn.close()
#     except Exception as e:
#         print("⚠️ Could not log visit:", e)

#     return render_template('patient_details.html', data=data)


# # ===============================
# # 🔑 FORGOT PASSWORD
# # ===============================
# @patient_bp.route('/forgot_password', methods=['GET', 'POST'])
# def forgot_password():
#     if request.method == 'POST':
#         email = request.form['email'].strip().lower()

#         conn = get_db()
#         user = conn.execute("SELECT id, name FROM users WHERE email=?", (email,)).fetchone()
#         conn.close()

#         if user:
#             token = serializer.dumps(email, salt='password-reset-salt')
#             reset_url = url_for('patient.reset_password', token=token, _external=True)
#             send_reset_email(user[1], email, reset_url)
#             flash("📧 A password reset link has been sent to your email.", "info")
#         else:
#             flash("❌ No account found with that email.", "danger")

#     return render_template('forgot_password.html')


# # ===============================
# # 🔄 RESET PASSWORD
# # ===============================
# @patient_bp.route('/reset_password/<token>', methods=['GET', 'POST'])
# def reset_password(token):
#     try:
#         email = serializer.loads(token, salt='password-reset-salt', max_age=600)
#     except (SignatureExpired, BadSignature):
#         flash("⚠️ Invalid or expired reset link.", "danger")
#         return redirect(url_for('patient.forgot_password'))

#     if request.method == 'POST':
#         new_password = request.form['password']
#         hashed = generate_password_hash(new_password)

#         conn = get_db()
#         conn.execute("UPDATE users SET password_hash=? WHERE email=?", (hashed, email))
#         conn.commit()
#         conn.close()

#         flash("✅ Password has been reset successfully!", "success")
#         return redirect(url_for('patient.login_page'))

#     return render_template('reset_password.html', email=email)









from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import check_password_hash, generate_password_hash
import sqlite3, cv2, os, numpy as np, base64, random, dlib
from datetime import datetime
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from email_utils import send_reset_email  # make sure this file exists and works

# === Blueprint ===
patient_bp = Blueprint('patient', __name__, template_folder='../templates')

# === Paths ===
RECOGNIZER_PATH = os.path.join("recognizer", "trainingdata.yml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === Dlib Detector (Optional smoother detection) ===
detector = dlib.get_frontal_face_detector()
sp = None
try:
    sp_path = os.path.join("recognizer", "shape_predictor_5_face_landmarks.dat")
    if os.path.exists(sp_path):
        sp = dlib.shape_predictor(sp_path)
except Exception:
    sp = None

# === Security Token Setup ===
SECRET_KEY = "caresync_secret_key"
serializer = URLSafeTimedSerializer(SECRET_KEY)

# === Database Connection ===
def get_db():
    # ✅ Use your absolute database path
    db_path = r"C:\Users\subod\OneDrive\Desktop\careSync\Face Detection\backend\database.db"
    return sqlite3.connect(db_path)


# ===============================
# 🧑‍💻 PATIENT LOGIN
# ===============================
@patient_bp.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']

        conn = get_db()
        user = conn.execute(
            "SELECT id, name, email, password_hash FROM users WHERE email=?",
            (email,)
        ).fetchone()
        conn.close()

        if user and user[3] and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['user_name'] = user[1]
            flash(f"Welcome {user[1]}! Please verify your face.", "info")
            return redirect(url_for('patient.face_verify_page'))
        else:
            flash("❌ Invalid email or password.", "danger")

    return render_template('patient_login.html')


# ===============================
# 🧠 FACE VERIFICATION PAGE
# ===============================
@patient_bp.route('/verify', methods=['GET'])
def face_verify_page():
    if 'user_id' not in session:
        flash("Please login first.", "warning")
        return redirect(url_for('patient.login_page'))
    return render_template('verify_face.html', user_id=session['user_id'], name=session['user_name'])


# ===============================
# ⚡ FACE VERIFICATION (AJAX)
# ===============================
@patient_bp.route('/verify_frame', methods=['POST'])
def verify_frame():
    data = request.get_json()
    img_data = data.get('image')
    user_id = int(data.get('user_id'))

    # Check if model exists
    if not os.path.exists(RECOGNIZER_PATH):
        return jsonify({"status": "error", "message": "⚠️ Face model not trained yet."})

    # Decode base64 image
    try:
        img_bytes = base64.b64decode(img_data.split(',')[1])
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    except Exception as e:
        print("❌ Image decode error:", e)
        return jsonify({"status": "error", "message": "Failed to decode image"})

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(90, 90))
    if len(faces) == 0:
        return jsonify({"status": "no_face"})

    # Load trained recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(RECOGNIZER_PATH)

    confidences = []
    predictions = []

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue

        face_resized = cv2.resize(face_roi, (200, 200))
        pred_id, conf = recognizer.predict(face_resized)
        predictions.append(pred_id)
        confidences.append(conf)

    if not confidences:
        return jsonify({"status": "no_face"})

    avg_conf = sum(confidences) / len(confidences)
    most_common_id = max(set(predictions), key=predictions.count)

    print(f"🧩 Frame analysis → PredID={most_common_id}, AvgConf={avg_conf:.2f}, ExpectedID={user_id}")

    # ✅ Decision logic
    if most_common_id == user_id and avg_conf < 75:
        session['verified'] = True
        print(f"✅ Verified: ID={most_common_id}, Confidence={avg_conf:.2f}")
        return jsonify({"status": "verified"})
    elif most_common_id == user_id and avg_conf < 95:
        session['verified'] = True
        print(f"⚠️ Borderline but accepted: ID={most_common_id}, Conf={avg_conf:.2f}")
        return jsonify({"status": "verified"})
    else:
        print(f"❌ Failed verification → Pred={most_common_id}, Conf={avg_conf:.2f}")
        return jsonify({"status": "failed"})


# ===============================
# 📋 PATIENT DETAILS PAGE
# ===============================
@patient_bp.route('/details')
def details():
    if 'user_id' not in session:
        flash("Please login first.", "warning")
        return redirect(url_for('patient.login_page'))

    if not session.get('verified'):
        flash("Please verify your face first.", "warning")
        return redirect(url_for('patient.face_verify_page'))

    conn = get_db()
    user = conn.execute("""
        SELECT name, email, problem, appointment_date
        FROM users WHERE id=?
    """, (session['user_id'],)).fetchone()
    conn.close()

    if not user:
        flash("User not found!", "danger")
        return redirect(url_for('patient.login_page'))

    doctors = [
        "Dr. John Smith", "Dr. Jennifer Williams", "Dr. Michael Brown",
        "Dr. Sarah Davis", "Dr. Robert Miller"
    ]
    time_slots = ["09:00 AM", "10:30 AM", "01:00 PM", "03:30 PM", "05:00 PM"]

    data = {
        "name": user[0],
        "email": user[1],
        "problem": user[2],
        "doctor": random.choice(doctors),
        "date": user[3] or datetime.now().strftime("%Y-%m-%d"),
        "time": random.choice(time_slots),
        "status": "booked"
    }

    # Log visit
    try:
        conn = get_db()
        conn.execute("""
            INSERT INTO visits (user_id, checkin_time, recognized)
            VALUES (?, datetime('now'), 1)
        """, (session['user_id'],))
        conn.commit()
        conn.close()
    except Exception as e:
        print("⚠️ Could not log visit:", e)

    return render_template('patient_details.html', data=data)


# ===============================
# 🔑 FORGOT PASSWORD
# ===============================
@patient_bp.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()

        conn = get_db()
        user = conn.execute("SELECT id, name FROM users WHERE email=?", (email,)).fetchone()
        conn.close()

        if user:
            token = serializer.dumps(email, salt='password-reset-salt')
            reset_url = url_for('patient.reset_password', token=token, _external=True)
            send_reset_email(user[1], email, reset_url)
            flash("📧 A password reset link has been sent to your email.", "info")
        else:
            flash("❌ No account found with that email.", "danger")

    return render_template('forgot_password.html')


# ===============================
# 🔄 RESET PASSWORD
# ===============================
@patient_bp.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = serializer.loads(token, salt='password-reset-salt', max_age=600)
    except (SignatureExpired, BadSignature):
        flash("⚠️ Invalid or expired reset link.", "danger")
        return redirect(url_for('patient.forgot_password'))

    if request.method == 'POST':
        new_password = request.form['password']
        hashed = generate_password_hash(new_password)

        conn = get_db()
        conn.execute("UPDATE users SET password_hash=? WHERE email=?", (hashed, email))
        conn.commit()
        conn.close()

        flash("✅ Password has been reset successfully!", "success")
        return redirect(url_for('patient.login_page'))

    return render_template('reset_password.html', email=email)
