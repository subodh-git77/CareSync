# 🏥 CareSync  
### AI-Powered Hospital Appointment & Face Recognition System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Flask-Web%20Framework-green?style=for-the-badge&logo=flask">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge">
</p>

<p align="center">
  🚀 <b>CareSync</b> is a smart healthcare platform that combines <b>AI face recognition</b> with <b>hospital appointment scheduling</b> for secure, touchless patient verification.
</p>

---

## ✨ Overview

**CareSync** is a full-stack hospital management system that allows patients to:

- 📅 Book hospital appointments online  
- 👤 Verify identity using **AI-based face recognition**  
- 🏥 Perform **touchless hospital check-ins**  
- 🔐 Access appointment history securely  

The system is optimized for **desktop and mobile browsers** and follows a **modular Flask architecture** for scalability and maintainability.

---

## 🧠 Key Features

### 👤 Patient Module
- Secure user registration and login
- Facial data capture and AI-based verification
- Online appointment booking & schedule tracking
- Face-based hospital check-in
- Email-based password recovery

### 🛡️ Admin Module
- Secure admin authentication
- Patient verification management
- Appointment monitoring
- Visit logs and system insights

### 🤖 AI & Security
- LBPH Face Recognition (OpenCV)
- Encrypted password storage
- Session-based authentication
- Token-based password reset system

---

## 🛠️ Tech Stack

| Category | Technologies |
|--------|-------------|
| **Backend** | Python, Flask |
| **Database** | SQLite |
| **AI / Computer Vision** | OpenCV, LBPH, Dlib |
| **Frontend** | HTML, Tailwind CSS, JavaScript |
| **Authentication & Security** | Werkzeug, Sessions, Tokens |
| **Tools** | Git, GitHub |

---

## 📁 Project Structure

```text
Face-Detection/
│
├── backend/
│   ├── app.py
│   ├── database.db
│   ├── models.py
│   ├── face_utils.py
│   ├── email_utils.py
│   ├── routes/
│   │   ├── auth_routes.py
│   │   ├── patient_routes.py
│   │   ├── admin_routes.py
│   │   ├── appointment_routes.py
│   │   └── detect_routes.py
│   ├── templates/
│   ├── static/
│   └── recognizer/
│       └── trainingdata.yml
│
├── dataset/
├── requirements.txt
└── README.md
```
## 🚀 How It Works

1️⃣ Patient registers and uploads facial data  
2️⃣ System captures and stores face images securely  
3️⃣ AI model is trained using LBPH face recognition  
4️⃣ Patient logs in using email and password  
5️⃣ Face verification confirms patient identity  
6️⃣ Patient books, views, and manages appointments  
7️⃣ Touchless face-based check-in at the hospital  

---

## 📷 Face Recognition Flow

- Uses **LBPH (Local Binary Patterns Histogram)** algorithm  
- Real-time face detection via webcam or mobile camera  
- Face features converted into histogram patterns  
- Confidence score used for identity verification  
- Optimized for **Chrome browser (Desktop & Mobile)**  

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/subodh-git77/caresync
```
### 2️⃣ Navigate to Project Directory
```bash
cd Face-Detection/backend
```
### 3️⃣ Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate      # On Linux/Mac
venv\Scripts\activate         # On Windows
```
### 4️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 5️⃣ Run the Application
```bash
python app.py
```
### 6️⃣ Access the Application
```
http://127.0.0.1:5000
```
## 📈 Future Enhancements

👨‍⚕️ Role-based doctor and staff dashboards

📩 SMS & email appointment reminders

☁️ Cloud database support (PostgreSQL / Firebase)

🧠 Deep learning face recognition (CNN / FaceNet)

🐳 Docker containerization & cloud deployment

📊 Analytics dashboard for hospital management

## 👨‍💻 Author

Subodh Kumar Agrahari

📧 Email: subodhagrahari717@gmail.com

💼 LinkedIn: https://linkedin.com/in/subodh-kumar-agrahari-0449652a9/

🌐 GitHub: https://github.com/subodh-git77
