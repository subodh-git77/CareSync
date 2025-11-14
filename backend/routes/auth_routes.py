# routes/auth_routes.py
from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
from email_utils import send_appointment_email

# Define Blueprint
auth = Blueprint('auth', __name__, template_folder='../templates')

# === Use absolute database path ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "../database.db")

def get_db():
    """Connect to the main database."""
    return sqlite3.connect(DB_PATH)

# === REGISTER ROUTE ===
@auth.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            id = int(request.form['id'])
            name = request.form['name']
            email = request.form['email']
            password = request.form['password']
            phone = request.form.get('phone')
            address = request.form.get('address')
            problem = request.form.get('problem')
            appointment = request.form.get('appointment')

            # Hash password securely
            pw_hash = generate_password_hash(password)

            # Save to database
            conn = get_db()
            conn.execute("""
                INSERT OR REPLACE INTO users (id, name, email, phone, address, password_hash, problem, appointment_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (id, name, email, phone, address, pw_hash, problem, appointment))
            conn.commit()
            conn.close()

            # Send confirmation email if appointment is provided
            if appointment:
                send_appointment_email(name, email, appointment, problem)

            flash('✅ Account created successfully! Please register your face next.', 'success')
            return redirect(url_for('register.register_page'))  # Move to face registration
        except Exception as e:
            flash(f"❌ Error: {e}", 'danger')
            return redirect(url_for('auth.register'))

    return render_template('auth/register.html')


# === LOGIN ROUTE ===
@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db()
        row = conn.execute("SELECT id, password_hash, name FROM users WHERE email=?", (email,)).fetchone()
        conn.close()

        if row and check_password_hash(row[1], password):
            session['user_id'] = row[0]
            session['user_name'] = row[2]
            flash(f"Welcome back, {row[2]}!", 'success')
            return redirect(url_for('detect.detect_page'))  # Redirect to face check-in or dashboard
        else:
            flash("❌ Invalid credentials. Please try again.", 'danger')

    return render_template('auth/login.html')


# === LOGOUT ROUTE ===
@auth.route('/logout')
def logout():
    session.clear()
    flash("You’ve been logged out successfully.", 'info')
    return redirect(url_for('home'))
