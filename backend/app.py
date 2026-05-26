# ===============================
# CareSync - Smart Hospital Appointments
# ===============================

from flask import Flask, render_template, redirect, url_for
from dotenv import load_dotenv
import os
import socket

from database import init_db, ensure_approved_column

# === Load environment variables ===
load_dotenv()

# === Initialize Flask app ===
app = Flask(__name__, static_folder="static", template_folder="templates")

# === Secure Flask Secret Key ===
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default_secret_key")

# === Ensure required folders exist ===
os.makedirs("dataset", exist_ok=True)
os.makedirs("recognizer", exist_ok=True)
os.makedirs("backend/static/images", exist_ok=True)

# === Initialize Database ===
init_db()
ensure_approved_column()

# === Import Blueprints ===
from routes.register_routes import register_bp
from routes.train_routes import train_bp
from routes.detect_routes import detect_bp
from routes.admin_routes import admin_bp
from routes.patient_routes import patient_bp

# === Register Blueprints ===
app.register_blueprint(register_bp, url_prefix="/register")
app.register_blueprint(train_bp, url_prefix="/train")
app.register_blueprint(detect_bp, url_prefix="/detect")
app.register_blueprint(admin_bp, url_prefix="/admin")
app.register_blueprint(patient_bp, url_prefix="/patient")

# ===============================
# HOME ROUTES
# ===============================
@app.route("/")
def home():
    """Main landing page"""
    return render_template("index.html")


@app.route("/features")
def features():
    """Show CareSync features page"""
    return render_template("features.html")


@app.route("/login")
def login_redirect():
    """Redirect to patient login"""
    return redirect(url_for("patient.login_page"))


# ===============================
# ERROR HANDLERS
# ===============================
@app.errorhandler(404)
def not_found(e):
    """Custom 404 Page"""
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(e):
    """Custom 500 Page"""
    return render_template("500.html", error=str(e)), 500


# ===============================
# START SERVER
# ===============================
if __name__ == "__main__":
    print("Starting CareSync Flask Server...")

    port = int(os.getenv("PORT", 5000))

    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)

        print(f"Localhost: http://127.0.0.1:{port}/")
        print(f"Mobile Link: http://{local_ip}:{port}/")
        print("Make sure your phone and laptop are connected to the same Wi-Fi network.")
    except Exception:
        print(f"Localhost: http://127.0.0.1:{port}/")

    app.run(host="0.0.0.0", port=port, debug=False)
