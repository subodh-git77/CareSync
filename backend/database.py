import sqlite3
import os
import shutil
from datetime import datetime

# === Base Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database.db")
BACKUP_DIR = os.path.join(BASE_DIR, "backups")

# === Ensure Backup Folder Exists ===
os.makedirs(BACKUP_DIR, exist_ok=True)


def backup_database():
    """Create a timestamped backup of the current database."""
    if os.path.exists(DB_PATH):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(BACKUP_DIR, f"database_backup_{timestamp}.db")
        shutil.copy2(DB_PATH, backup_file)
        print(f"🗄️  Backup created: {backup_file}")
    else:
        print("⚠️ No database file found — skipping backup.")


def init_db():
    """Initialize database tables safely."""
    # Always create a backup first (if DB exists)
    backup_database()

    conn = sqlite3.connect(DB_PATH)

    # === USERS TABLE (Patients) ===
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER,
        gender TEXT,
        email TEXT UNIQUE,
        phone TEXT,
        address TEXT,
        password_hash TEXT,
        problem TEXT,
        appointment_date TEXT
    );
    """)

    # === ADMINS TABLE ===
    conn.execute("""
    CREATE TABLE IF NOT EXISTS admins (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        password_hash TEXT
    );
    """)

    # === APPOINTMENTS TABLE ===
    conn.execute("""
    CREATE TABLE IF NOT EXISTS appointments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        doctor TEXT,
        date TEXT,
        time_slot TEXT,
        symptoms TEXT,
        status TEXT DEFAULT 'booked',
        created_at TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    """)

    # === VISITS TABLE (for face check-ins) ===
    conn.execute("""
    CREATE TABLE IF NOT EXISTS visits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        appointment_id INTEGER,
        user_id INTEGER,
        checkin_time TEXT,
        recognized INTEGER,
        FOREIGN KEY(appointment_id) REFERENCES appointments(id),
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    """)

    conn.commit()
    conn.close()
    print("✅ Database initialized successfully (users, admins, appointments, visits).")


def ensure_approved_column():
    """Ensure that the 'approved' column exists in the users table."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("PRAGMA table_info(users)")
    cols = [r[1].lower() for r in cur.fetchall()]

    if "approved" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN approved INTEGER DEFAULT 0;")
        conn.commit()
        print("✅ Added 'approved' column to users table.")
    else:
        print("ℹ️ 'approved' column already exists — skipping alter.")

    conn.close()


# === Entry Point ===
if __name__ == "__main__":
    print("🚀 Initializing CareSync Database...")
    init_db()
    ensure_approved_column()
    print("✅ All setup complete.")
