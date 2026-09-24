"""
SonicSentinel AI - NextWave Acoustic Intelligence Web Application
Backend Server using Python Flask and SQLite Authentication
"""

import os
import csv
import io
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response
)
import database

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize SQLite database schema and seed users
database.init_db()


def login_required(f):
    """Decorator to ensure route is only accessible to authenticated sessions"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please sign in with your credentials to access the secure monitoring console.", "warning")
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function


@app.context_processor
def inject_user_and_context():
    """Inject current user information and metadata into all templates"""
    current_user = None
    if 'user_id' in session:
        current_user = database.get_user_by_id(session['user_id'])
    
    return {
        'current_user': current_user,
        'app_name': 'SonicSentinel AI',
        'system_version': '1.0.0-Enterprise',
        'system_uptime': '99.94%'
    }


# ==========================================
# AUTHENTICATION ROUTES
# ==========================================

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        identifier = request.form.get('identifier', '').strip()
        password = request.form.get('password', '').strip()
        remember = request.form.get('remember') == 'on'

        if not identifier or not password:
            flash("Please enter both username/email and password.", "error")
            return render_template('login.html', identifier=identifier)

        user, error = database.verify_user_credentials(identifier, password)
        if error:
            flash(error, "error")
            return render_template('login.html', identifier=identifier)

        # Successful login
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['role'] = user['role']
        session['full_name'] = user['full_name']
        session.permanent = remember

        flash(f"Welcome back, {user['full_name']}.", "success")
        next_page = request.args.get('next')
        return redirect(next_page or url_for('dashboard'))

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        full_name = request.form.get('full_name', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        role = request.form.get('role', 'Security Operator')
        station = request.form.get('station', 'Terminal #04 (Sector B)')

        # Validations
        if not username or not email or not full_name or not password:
            flash("All required fields must be completed.", "error")
            return render_template('register.html', username=username, email=email, full_name=full_name, role=role)

        if len(password) < 6:
            flash("Password must be at least 6 characters in length.", "error")
            return render_template('register.html', username=username, email=email, full_name=full_name, role=role)

        if password != confirm_password:
            flash("Passwords do not match. Please re-enter.", "error")
            return render_template('register.html', username=username, email=email, full_name=full_name, role=role)

        success, result = database.create_user(
            username=username,
            email=email,
            password=password,
            full_name=full_name,
            role=role,
            station=station
        )

        if not success:
            flash(result, "error")
            return render_template('register.html', username=username, email=email, full_name=full_name, role=role)

        flash("Registration successful! You can now log in with your credentials.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    flash("You have been securely logged out of SonicSentinel AI.", "info")
    return redirect(url_for('login'))


# ==========================================
# PROTECTED APPLICATION MODULES
# ==========================================

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', active_page='dashboard')


@app.route('/live-monitoring')
@login_required
def live_monitoring():
    return render_template('live_monitoring.html', active_page='live_monitoring')


@app.route('/audio-upload')
@login_required
def audio_upload():
    return render_template('audio_upload.html', active_page='audio_upload')


@app.route('/audio-analysis')
@login_required
def audio_analysis():
    return render_template('audio_analysis.html', active_page='audio_analysis')


@app.route('/alerts')
@login_required
def alerts():
    return render_template('alerts.html', active_page='alerts')


@app.route('/manual-review')
@login_required
def manual_review():
    return render_template('manual_review.html', active_page='manual_review')


@app.route('/events')
@login_required
def event_history():
    return render_template('event_history.html', active_page='events')


@app.route('/analytics')
@login_required
def analytics():
    return render_template('analytics.html', active_page='analytics')


@app.route('/reports')
@login_required
def reports():
    return render_template('reports.html', active_page='reports')


@app.route('/settings')
@login_required
def settings():
    return render_template('settings.html', active_page='settings')


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = database.get_user_by_id(session['user_id'])
    if request.method == 'POST':
        full_name = request.form.get('full_name', user['full_name']).strip()
        station = request.form.get('station', user['station']).strip()
        theme = request.form.get('theme', user['theme_preference'])

        database.update_user_profile(user['id'], full_name, station, theme)
        session['full_name'] = full_name
        flash("Your profile preferences have been successfully updated.", "success")
        return redirect(url_for('profile'))

    return render_template('profile.html', user=user, active_page='profile')


@app.route('/export-csv')
@login_required
def export_csv():
    """Export standard events to CSV file"""
    output = io.StringIO()
    writer = csv.writer(output)
    
    headers = [
        "Audio_ID", "Filename", "Category", "Python_Confidence",
        "GTM_Confidence", "Confidence_Delta", "Audio_Quality",
        "Severity", "Status", "Timestamp"
    ]
    writer.writerow(headers)
    
    # Mock data rows
    sample_rows = [
        ["AUD-2026-8801", "facility_east_wing_gunshot_01.wav", "Gunshot", "96.4%", "94.8%", "1.6%", "Good", "Critical", "Active", "2026-09-24 08:12:45"],
        ["AUD-2026-8802", "emergency_stairwell_scream_03.flac", "Panic Scream", "91.2%", "89.7%", "1.5%", "Good", "Critical", "Escalated", "2026-09-24 08:14:10"],
        ["AUD-2026-8803", "loading_bay_help_call_02.wav", "Person Asking for Help", "88.5%", "87.0%", "1.5%", "Acceptable", "Critical", "Acknowledged", "2026-09-24 08:18:22"],
        ["AUD-2026-8804", "warehouse_window_shatter.mp3", "Glass Breaking", "94.0%", "92.5%", "1.5%", "Good", "High", "Reviewed", "2026-09-24 07:45:00"],
        ["AUD-2026-8805", "generator_bearing_cavitation.wav", "Machinery Fault", "95.8%", "93.1%", "2.7%", "Good", "High", "Reviewed", "2026-09-24 06:30:15"],
        ["AUD-2026-8806", "fire_evacuation_strobe_siren.wav", "Alarm or Siren", "98.2%", "97.5%", "0.7%", "Good", "High", "Closed", "2026-09-24 05:10:00"],
        ["AUD-2026-8807", "loading_zone_ambiguous_echo.wav", "Vehicle Horn", "54.2%", "61.8%", "7.6%", "Poor", "Medium", "Manual Review", "2026-09-24 08:05:12"],
        ["AUD-2026-8808", "hallway_shouting_dispute.wav", "Aggression", "64.5%", "61.0%", "3.5%", "Acceptable", "High", "Manual Review", "2026-09-24 07:58:30"],
        ["AUD-2026-8809", "perimeter_fence_canine_bark.mp3", "Animal Sound", "93.4%", "91.8%", "1.6%", "Good", "Low", "Closed", "2026-09-24 07:15:40"],
        ["AUD-2026-8810", "hvac_ventilation_ambient_08.ogg", "Background Noise", "97.1%", "96.0%", "1.1%", "Good", "Informational", "Closed", "2026-09-24 06:00:22"]
    ]
    
    for row in sample_rows:
        writer.writerow(row)
        
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=SonicSentinel_Event_Audit.csv"}
    )


if __name__ == '__main__':
    print("==========================================================")
    print("  SonicSentinel AI - NextWave Acoustic Intelligence")
    print("  Enterprise Acoustic Threat Detection & Monitoring System")
    print("  Server running on http://127.0.0.1:5000")
    print("  Default Login:")
    print("    Username: operator  |  Password: Operator@123")
    print("    Username: admin     |  Password: Admin@123")
    print("==========================================================")
    app.run(host='127.0.0.1', port=5000, debug=True)
