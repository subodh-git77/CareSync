import sqlite3
from werkzeug.security import generate_password_hash

# Connect to your database
conn = sqlite3.connect("database.db")

# === User details ===
email = "ummauryas2005@gmail.com"  # <-- your registered email
new_password = "newpass123"         # You can change this to any password you want

# === Generate a secure hash ===
pw_hash = generate_password_hash(new_password)

# === Update password in the database ===
conn.execute("UPDATE users SET password_hash=? WHERE email=?", (pw_hash, email))
conn.commit()
conn.close()

print(f"✅ Password updated successfully for {email}!")
print(f"👉 New password: {new_password}")
