# routes/appointment_routes.py
from flask import Blueprint, render_template, request, redirect, url_for, session
import sqlite3, datetime
from email_utils import send_appointment_email

appointment_bp = Blueprint('appointment', __name__, template_folder='../templates')

def get_db(): return sqlite3.connect('database.db')

@appointment_bp.route('/book', methods=['GET','POST'])
def book():
    # ensure user logged in
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('auth.login'))

    if request.method == 'POST':
        doctor = request.form['doctor']
        date = request.form['date']  # 'YYYY-MM-DD'
        time_slot = request.form['time_slot']
        symptoms = request.form['symptoms']
        created_at = datetime.datetime.utcnow().isoformat()

        conn = get_db()
        cur = conn.cursor()
        cur.execute('INSERT INTO appointments (user_id,doctor,date,time_slot,symptoms,created_at) VALUES (?,?,?,?,?,?)',
                    (user_id, doctor, date, time_slot, symptoms, created_at))
        conn.commit()
        appointment_id = cur.lastrowid
        # fetch user details email/name
        user = conn.execute('SELECT name,email FROM users WHERE id=?', (user_id,)).fetchone()
        if user:
            send_appointment_email(user[0], user[1], date, symptoms, time_slot)
        conn.close()
        return redirect(url_for('appointment.myappointments'))
    # GET: show booking form; you can send sample doctor list
    doctors = ['Dr. A', 'Dr. B', 'Dr. C']
    return render_template('patient/book.html', doctors=doctors)

@appointment_bp.route('/myappointments')
def myappointments():
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('auth.login'))
    conn = get_db()
    rows = conn.execute('SELECT * FROM appointments WHERE user_id=? ORDER BY date DESC', (user_id,)).fetchall()
    conn.close()
    # categorize past/upcoming/today
    return render_template('patient/myappointments.html', appointments=rows)


