import sqlite3
import os
from werkzeug.security import generate_password_hash

# === Get the absolute database path ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")

# === Connect to the main database ===
conn = sqlite3.connect(DB_PATH)

# === Ensure the admins table exists ===
conn.execute("""
CREATE TABLE IF NOT EXISTS admins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL
);
""")

# === Admin details (Default admin) ===
admin_name = "Dr. Admin"
admin_email = "admin@caresync.com"
admin_password = "admin123"  # Change this before deployment

# === Hash the password ===
password_hash = generate_password_hash(admin_password)

# === Check for existing admin ===
cursor = conn.execute("SELECT id FROM admins WHERE email = ?", (admin_email,))
existing_admin = cursor.fetchone()

if existing_admin:
    print(f"⚠️ Admin with email '{admin_email}' already exists (ID: {existing_admin[0]}).")
else:
    conn.execute("""
        INSERT INTO admins (name, email, password_hash)
        VALUES (?, ?, ?)
    """, (admin_name, admin_email, password_hash))
    conn.commit()
    print("✅ Admin account created successfully!")
    print(f"   👤 Name: {admin_name}")
    print(f"   📧 Email: {admin_email}")
    print(f"   🔑 Password: {admin_password}")

conn.close()
