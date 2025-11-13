import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# ==========================
# 🔐 Load environment variables
# ==========================
load_dotenv()

SENDER_EMAIL = os.getenv("EMAIL_USER")
SENDER_PASSWORD = os.getenv("EMAIL_PASS")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587


# ==========================
# 📤 Generic Email Sender
# ==========================
def send_email(to_email: str, subject: str, html_body: str):
    """Reusable function to send HTML email via Gmail securely."""
    try:
        if not SENDER_EMAIL or not SENDER_PASSWORD:
            print("❌ Missing EMAIL_USER or EMAIL_PASS in .env file.")
            return False

        msg = MIMEMultipart("alternative")
        msg["From"] = SENDER_EMAIL
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        print(f"✅ Email sent successfully to {to_email}")
        return True

    except Exception as e:
        print(f"❌ Failed to send email to {to_email}: {e}")
        return False


# ==========================
# 📅 Appointment Confirmation
# ==========================
def send_appointment_email(name: str, to_email: str, appointment_date: str, problem: str):
    subject = "📅 Appointment Confirmation - CareSync"
    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; color: #333;">
        <h2>Hi {name},</h2>
        <p>Your appointment has been successfully scheduled with <b>CareSync Hospital</b>.</p>
        <ul>
            <li><b>🩺 Problem:</b> {problem}</li>
            <li><b>📅 Date:</b> {appointment_date}</li>
            <li><b>⏰ Time:</b> 10:00 AM</li>
        </ul>
        <p>Please bring your ID card and arrive 10 minutes early.</p>
        <p>Thank you for choosing <b>CareSync</b> for your healthcare needs!</p>
        <hr>
        <p>— The CareSync Team 🏥</p>
    </body>
    </html>
    """
    send_email(to_email, subject, html_body)


# ==========================
# ✅ Face Verification Success
# ==========================
def send_confirmation_email(to_email: str, name: str):
    subject = "✅ Face Verification Successful - CareSync"
    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; color: #333;">
        <h2>Hello {name},</h2>
        <p>We’ve successfully verified your face using <b>CareSync’s AI Recognition System</b>.</p>
        <p>Thank you for visiting today. Your verification is complete.</p>
        <p>We appreciate your time and cooperation.</p>
        <hr>
        <p>Best regards,<br><b>CareSync Hospital Team 🏥</b></p>
    </body>
    </html>
    """
    send_email(to_email, subject, html_body)


# ==========================
# 🔑 Password Reset Email
# ==========================
def send_reset_email(name: str, recipient_email: str, reset_link: str):
    subject = "🔑 Password Reset Request - CareSync"
    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; color: #333;">
        <h2>Hi {name},</h2>
        <p>We received a request to reset your CareSync account password.</p>
        <p>Please click the button below to reset it:</p>
        <p style="margin: 20px 0;">
            <a href="{reset_link}" 
               style="background-color:#007bff;color:white;padding:12px 25px;
               text-decoration:none;border-radius:8px;font-weight:bold;">
                Reset My Password
            </a>
        </p>
        <p>This link will expire in <b>10 minutes</b> for your security.</p>
        <p>If you didn’t request this, you can ignore this email.</p>
        <hr>
        <p>— The CareSync Security Team 🔐</p>
    </body>
    </html>
    """
    send_email(recipient_email, subject, html_body)
