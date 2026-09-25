"""
SonicSentinel AI - NextWave Acoustic Intelligence Web Application
Backend Server using Python Flask and SQLite Authentication
"""

import os
import csv
import io
import json
import uuid
from datetime import datetime
from functools import wraps
from werkzeug.utils import secure_filename
from flask import (
    Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response, abort, send_file
)
import database
from src.utils.audio_hash import compute_audio_hash
from src.services.quality_assessor import assess_audio_quality
from src.services.feature_extractor import extract_spectral_features
from src.services.audio_dsp import (
    load_audio_file, parse_pcm_buffer, render_waveform_image, render_spectrogram_image
)
from src.services.ml_service import classify_audio_dual
from src.services.alert_engine import evaluate_alert, reload_rules
from src.services.pdf_generator import generate_incident_pdf
from config.settings import UPLOAD_FOLDER

app = Flask(__name__)
# Cryptographically signed sessions
app.secret_key = os.environ.get('SECRET_KEY', 'sonicsentinel-secure-session-key-2026-acousticx')
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Initialize SQLite database schema and seed users
database.init_db()


# ==========================================
# SECURITY HEADERS & BROWSER CACHE CONTROL
# ==========================================

@app.after_request
def add_security_headers(response):
    """
    Prevents browser back button from displaying cached protected pages after logout.
    Forces HTTP revalidation on every navigation.
    """
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    return response


# ==========================================
# AUTHENTICATION & RBAC DECORATORS
# ==========================================

def get_role_dashboard_url(role=None):
    """Returns the dedicated dashboard URL for a given role"""
    if not role:
        role = session.get('role', database.ROLE_NORMAL_USER)

    normalized = str(role).strip().lower().replace('_', ' ')
    if 'admin' in normalized:
        return url_for('admin_dashboard')
    elif 'security' in normalized or 'operator' in normalized and 'maintenance' not in normalized:
        return url_for('operator_dashboard')
    elif 'review' in normalized:
        return url_for('reviewer_dashboard')
    elif 'maint' in normalized:
        return url_for('maintenance_dashboard')
    else:
        return url_for('user_dashboard')


def login_required(f):
    """
    Ensures route is accessible only to active, authenticated sessions.
    Validates user against database on every request to prevent deactivated account access.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please sign in to access this protected module.", "warning")
            return redirect(url_for('login', next=request.url))

        # Query database to confirm user exists and is active
        user = database.get_user_by_id(session['user_id'])
        if not user:
            session.clear()
            flash("Your session is invalid. Please sign in again.", "error")
            return redirect(url_for('login'))

        if user['is_active'] == 0:
            session.clear()
            flash("Your account has been deactivated. Please contact an administrator.", "error")
            return redirect(url_for('login'))

        # Synchronize session state with database
        session['role'] = user['role']
        session['full_name'] = user['full_name']
        session['username'] = user['username']

        return f(*args, **kwargs)
    return decorated_function


def roles_required(*allowed_roles):
    """
    Strict server-side Role-Based Access Control (RBAC) decorator.
    Returns HTTP 403 Forbidden if user lacks required permissions.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                flash("Authentication required to access this resource.", "warning")
                return redirect(url_for('login', next=request.url))

            user = database.get_user_by_id(session['user_id'])
            if not user or user['is_active'] == 0:
                session.clear()
                flash("Your account is not active. Access denied.", "error")
                return redirect(url_for('login'))

            user_role = user['role']
            session['role'] = user_role

            # Normalize roles for robust comparison
            normalized_allowed = [r.strip().lower().replace('_', ' ') for r in allowed_roles]
            user_role_norm = user_role.strip().lower().replace('_', ' ')

            if user_role_norm not in normalized_allowed:
                return render_template(
                    '403.html',
                    user_role=user_role,
                    allowed_roles=allowed_roles,
                    target_endpoint=request.path
                ), 403

            return f(*args, **kwargs)
        return decorated_function
    return decorator


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
# PUBLIC AUTHENTICATION ROUTES
# ==========================================

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(get_role_dashboard_url())
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    # If user is already logged in, redirect directly to their own role dashboard
    if 'user_id' in session:
        return redirect(get_role_dashboard_url())

    if request.method == 'POST':
        identifier = request.form.get('identifier', '').strip()
        password = request.form.get('password', '').strip()
        remember = request.form.get('remember') == 'on'

        if not identifier or not password:
            flash("Please enter both your username/email and password.", "error")
            return render_template('login.html', identifier=identifier)

        user, error = database.verify_user_credentials(identifier, password)
        if error:
            flash(error, "error")
            return render_template('login.html', identifier=identifier)

        # Establish secure session
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['role'] = user['role']
        session['full_name'] = user['full_name']
        session.permanent = remember

        flash(f"Welcome back, {user['full_name']}.", "success")

        # Determine target dashboard
        next_page = request.args.get('next')
        # Avoid redirect loops to auth pages
        if next_page and not any(auth_path in next_page for auth_path in ['/login', '/register', '/logout']):
            return redirect(next_page)

        return redirect(get_role_dashboard_url(user['role']))

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    # If user is already logged in, redirect directly to their own role dashboard
    if 'user_id' in session:
        return redirect(get_role_dashboard_url())

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        full_name = request.form.get('full_name', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        # SECURITY RULE: Public register users ALWAYS become Normal User.
        # Frontend, URL, form data, or JSON role overrides are completely ignored.
        assigned_role = database.ROLE_NORMAL_USER

        # Validation
        if not username or not email or not full_name or not password:
            flash("All fields are required to register an account.", "error")
            return render_template('register.html', username=username, email=email, full_name=full_name)

        if len(username) < 3:
            flash("Username must be at least 3 characters in length.", "error")
            return render_template('register.html', username=username, email=email, full_name=full_name)

        # Block any attempt to register reserved administrator names
        if username.lower() in ['admin', 'administrator', 'root']:
            flash("The username 'admin' is a reserved system identifier.", "error")
            return render_template('register.html', username='', email=email, full_name=full_name)

        if len(password) < 6:
            flash("Password must be at least 6 characters in length.", "error")
            return render_template('register.html', username=username, email=email, full_name=full_name)

        if password != confirm_password:
            flash("Passwords do not match. Please re-enter.", "error")
            return render_template('register.html', username=username, email=email, full_name=full_name)

        success, result = database.create_user(
            username=username,
            email=email,
            password=password,
            full_name=full_name,
            role=assigned_role,
            station='Public Web Terminal'
        )

        if not success:
            flash(result, "error")
            return render_template('register.html', username=username, email=email, full_name=full_name)

        flash("Registration successful! Your Normal User account is ready. Please sign in.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if 'user_id' in session:
        return redirect(get_role_dashboard_url())

    if request.method == 'POST':
        identifier = request.form.get('identifier', '').strip()
        new_password = request.form.get('new_password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        if not identifier or not new_password or not confirm_password:
            flash("All fields are required to reset credentials.", "error")
            return render_template('forgot_password.html', identifier=identifier)

        if len(new_password) < 6:
            flash("New password must be at least 6 characters in length.", "error")
            return render_template('forgot_password.html', identifier=identifier)

        if new_password != confirm_password:
            flash("Passwords do not match. Please verify and try again.", "error")
            return render_template('forgot_password.html', identifier=identifier)

        success, message = database.reset_user_password(identifier, new_password)
        if not success:
            flash(message, "error")
            return render_template('forgot_password.html', identifier=identifier)

        flash(message, "success")
        return redirect(url_for('login'))

    return render_template('forgot_password.html')


@app.route('/logout')
def logout():
    session.clear()
    flash("You have been securely signed out. Cache and session credentials cleared.", "info")
    return redirect(url_for('login'))


# ==========================================
# GENERAL DASHBOARD ROUTE (AUTO-REDIRECT)
# ==========================================

@app.route('/dashboard')
@login_required
def dashboard():
    """Redirects authenticated user to their role-specific dashboard"""
    return redirect(get_role_dashboard_url())


# ==========================================
# DEDICATED ROLE DASHBOARDS
# ==========================================

@app.route('/admin/dashboard')
@login_required
@roles_required(database.ROLE_ADMINISTRATOR)
def admin_dashboard():
    """Administrator Control Center - Personnel Governance & System Policies"""
    users = database.get_all_users()
    return render_template('admin_dashboard.html', users=users, active_page='admin_dashboard')


@app.route('/admin/create-operator', methods=['POST'])
@login_required
@roles_required(database.ROLE_ADMINISTRATOR)
def admin_create_operator():
    """Allows Administrator to provision certified operator accounts"""
    full_name = request.form.get('full_name', '').strip()
    username = request.form.get('username', '').strip()
    email = request.form.get('email', '').strip()
    role = request.form.get('role', '').strip()
    station = request.form.get('station', 'Terminal #02 (East Wing)').strip()
    password = request.form.get('password', '').strip()

    # Validate role is strictly one of the operator creatable roles
    if role not in database.ADMIN_CREATABLE_ROLES:
        flash(f"Invalid operator role '{role}'. Permitted roles: {', '.join(database.ADMIN_CREATABLE_ROLES)}", "error")
        return redirect(url_for('admin_dashboard'))

    if not full_name or not username or not email or not password:
        flash("All fields are required to provision an operator account.", "error")
        return redirect(url_for('admin_dashboard'))

    if len(password) < 6:
        flash("Password must be at least 6 characters in length.", "error")
        return redirect(url_for('admin_dashboard'))

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
    else:
        flash(f"Operator account '{username}' successfully created with role '{role}'.", "success")

    return redirect(url_for('admin_dashboard'))


@app.route('/admin/toggle-user/<int:user_id>', methods=['POST'])
@login_required
@roles_required(database.ROLE_ADMINISTRATOR)
def admin_toggle_user(user_id):
    """Allows Administrator to activate or deactivate an account"""
    success, message = database.toggle_user_status(user_id)
    if success:
        flash(message, "success")
    else:
        flash(message, "error")
    return redirect(url_for('admin_dashboard'))


@app.route('/operator/dashboard')
@login_required
@roles_required(database.ROLE_SECURITY_OPERATOR, database.ROLE_ADMINISTRATOR)
def operator_dashboard():
    """Security Operator Command Console"""
    return render_template('operator_dashboard.html', active_page='operator_dashboard')


@app.route('/reviewer/dashboard')
@login_required
@roles_required(database.ROLE_AUDIO_REVIEWER, database.ROLE_ADMINISTRATOR)
def reviewer_dashboard():
    """Audio Reviewer Forensic Hub"""
    return render_template('reviewer_dashboard.html', active_page='reviewer_dashboard')


@app.route('/maintenance/dashboard')
@login_required
@roles_required(database.ROLE_MAINTENANCE_OPERATOR, database.ROLE_ADMINISTRATOR)
def maintenance_dashboard():
    """Maintenance Operator Sensor Diagnostics Console"""
    return render_template('maintenance_dashboard.html', active_page='maintenance_dashboard')


@app.route('/user/dashboard')
@login_required
@roles_required(database.ROLE_NORMAL_USER, database.ROLE_ADMINISTRATOR)
def user_dashboard():
    """Normal User Safety Portal"""
    return render_template('user_dashboard.html', active_page='user_dashboard')


# ==========================================
# MODULE ROUTES (STRICT RBAC ENFORCED)
# ==========================================

@app.route('/live-monitoring')
@login_required
@roles_required(database.ROLE_ADMINISTRATOR, database.ROLE_SECURITY_OPERATOR, database.ROLE_AUDIO_REVIEWER, database.ROLE_MAINTENANCE_OPERATOR)
def live_monitoring():
    return render_template('live_monitoring.html', active_page='live_monitoring')


@app.route('/audio-upload')
@login_required
def audio_upload():
    """Audio upload is accessible to all authenticated users"""
    return render_template('audio_upload.html', active_page='audio_upload')


@app.route('/audio-analysis')
@login_required
@roles_required(database.ROLE_ADMINISTRATOR, database.ROLE_SECURITY_OPERATOR, database.ROLE_AUDIO_REVIEWER)
def audio_analysis():
    return render_template('audio_analysis.html', active_page='audio_analysis')


@app.route('/alerts')
@login_required
@roles_required(database.ROLE_ADMINISTRATOR, database.ROLE_SECURITY_OPERATOR, database.ROLE_AUDIO_REVIEWER, database.ROLE_MAINTENANCE_OPERATOR)
def alerts():
    return render_template('alerts.html', active_page='alerts')


@app.route('/manual-review')
@login_required
@roles_required(database.ROLE_ADMINISTRATOR, database.ROLE_AUDIO_REVIEWER, database.ROLE_SECURITY_OPERATOR)
def manual_review():
    return render_template('manual_review.html', active_page='manual_review')


@app.route('/events')
@login_required
def event_history():
    """Event history is accessible to all authenticated users"""
    return render_template('event_history.html', active_page='events')


@app.route('/analytics')
@login_required
@roles_required(database.ROLE_ADMINISTRATOR, database.ROLE_SECURITY_OPERATOR)
def analytics():
    return render_template('analytics.html', active_page='analytics')


@app.route('/reports')
@login_required
@roles_required(database.ROLE_ADMINISTRATOR, database.ROLE_AUDIO_REVIEWER)
def reports():
    return render_template('reports.html', active_page='reports')


@app.route('/settings')
@login_required
@roles_required(database.ROLE_ADMINISTRATOR)
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

        # SECURITY RULE: User profile update NEVER alters user role
        database.update_user_profile(user['id'], full_name, station, theme)
        session['full_name'] = full_name
        flash("Your profile preferences have been successfully updated.", "success")
        return redirect(url_for('profile'))

    return render_template('profile.html', user=user, active_page='profile')


@app.route('/export-csv')
@login_required
def export_csv():
    """Export standard events from SQLite database to CSV file"""
    output = io.StringIO()
    writer = csv.writer(output)

    headers = [
        "Audio_ID", "Filename", "Category", "Python_Confidence",
        "GTM_Confidence", "Confidence_Delta", "Audio_Quality",
        "Severity", "Status", "Timestamp"
    ]
    writer.writerow(headers)

    events = database.get_all_audio_events()
    for e in events:
        writer.writerow([
            e['id'],
            e['original_filename'] or 'N/A',
            e['final_detected_class'] or e['python_predicted_class'] or 'N/A',
            f"{float(e['python_top_confidence'] or 0)*100:.1f}%",
            f"{float(e['gtm_top_confidence'] or 0)*100:.1f}%",
            f"{float(e['confidence_difference'] or 0)*100:.1f}%",
            e['quality_grade'] or 'Good',
            e['severity'] or 'Informational',
            e['alert_status'] or 'Active',
            str(e['created_at'])
        ])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=SonicSentinel_Event_Audit.csv"}
    )


# ==========================================
# ACOUSTIC REST & STREAMING API ENDPOINTS
# ==========================================

@app.route('/api/audio/upload', methods=['POST'])
@login_required
def api_audio_upload():
    """
    Ingests and analyzes single or batch audio files (SRS §6, §7, §8).
    Performs SHA-256 deduplication, audio quality check, feature extraction,
    dual AI classification, alert evaluation, and persistence.
    """
    if 'audio' not in request.files and 'file' not in request.files:
        return jsonify({"error": "No audio file provided in multipart payload"}), 400

    file = request.files.get('audio') or request.files.get('file')
    if not file or file.filename == '':
        return jsonify({"error": "Empty filename provided"}), 400

    filename = secure_filename(file.filename)
    audio_bytes = file.read()
    if len(audio_bytes) == 0:
        return jsonify({"error": "Uploaded file is 0 bytes"}), 400

    # 1. SHA-256 Fingerprint
    sha256_hash = compute_audio_hash(audio_bytes)

    # Check duplicate detection (AUD-10)
    conn = database.get_db_connection()
    existing_rec = conn.execute("SELECT * FROM audio_records WHERE sha256_hash = ?", (sha256_hash,)).fetchone()
    conn.close()

    audio_id = f"AUD-{uuid.uuid4().hex[:8].upper()}"
    saved_path = os.path.join(UPLOAD_FOLDER, f"{audio_id}_{filename}")
    with open(saved_path, 'wb') as f:
        f.write(audio_bytes)

    # 2. Audio DSP Loading & Resampling
    samples, sr, duration = load_audio_file(saved_path)

    # 3. Audio Quality Gatekeeper
    quality_result = assess_audio_quality(samples, sr)

    # 4. Feature Extraction
    features = extract_spectral_features(samples, sr)

    # 5. Dual AI Model Inference & Arbitration
    pred_result = classify_audio_dual(samples, features, quality_result, filename=filename)

    # If duplicate detected, flag consistency
    if existing_rec:
        pred_result["consistency_status"] = f"Duplicate Audio Hash (Matches {existing_rec['id']})"

    # 6. Alert Policy Evaluation
    alert_result = evaluate_alert(pred_result, session_id=f"user_{session.get('user_id', 'anon')}")

    # 7. Generate Waveform and Mel-Spectrogram Heatmap Images
    wave_url = render_waveform_image(samples, audio_id)
    spec_url = render_spectrogram_image(samples, audio_id)

    # 8. Persist to Database
    user_id = session.get('user_id')
    conn = database.get_db_connection()
    conn.execute('''
        INSERT INTO audio_records (
            id, user_id, source_type, file_path, original_filename,
            duration_seconds, sample_rate, channels, sha256_hash,
            quality_grade, is_clipped, is_silent, snr_db
        ) VALUES (?, ?, 'upload', ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
    ''', (
        audio_id, user_id, saved_path, filename,
        round(duration, 2), sr, sha256_hash,
        quality_result["quality_grade"],
        1 if quality_result["is_clipped"] else 0,
        1 if quality_result["is_silent"] else 0,
        quality_result["snr_db"]
    ))

    pred_id = f"PRED-{uuid.uuid4().hex[:8].upper()}"
    conn.execute('''
        INSERT INTO predictions (
            id, audio_id, python_predicted_class, python_top_confidence,
            python_probabilities, gtm_predicted_class, gtm_top_confidence,
            gtm_probabilities, confidence_difference, top_two_margin,
            consistency_status, final_detected_class, waveform_image_path,
            spectrogram_image_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        pred_id, audio_id,
        pred_result["python_class"], pred_result["python_confidence"],
        json.dumps(pred_result["python_probabilities"]),
        pred_result["gtm_class"], pred_result["gtm_confidence"],
        json.dumps(pred_result["gtm_probabilities"]),
        pred_result["confidence_difference"], pred_result["top_two_margin"],
        pred_result["consistency_status"], pred_result["final_class"],
        wave_url, spec_url
    ))

    alt_id = f"ALT-{uuid.uuid4().hex[:8].upper()}"
    conn.execute('''
        INSERT INTO alerts (
            id, audio_id, threat_category, severity, status,
            is_confirmed_repeated, recommended_action
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        alt_id, audio_id,
        alert_result["threat_category"], alert_result["severity"],
        alert_result["status"],
        1 if alert_result["is_confirmed_repeated"] else 0,
        alert_result["recommended_action"]
    ))

    if pred_result["consistency_status"] in ["Model Disagreement", "Uncertain Result"] or "Duplicate" in pred_result["consistency_status"]:
        rev_id = f"REV-{uuid.uuid4().hex[:8].upper()}"
        conn.execute('''
            INSERT INTO reviews (
                id, audio_id, triage_reason, original_decision,
                final_decision, is_override, reviewer_comments
            ) VALUES (?, ?, ?, ?, ?, 0, 'Pending forensic evaluation by Audio Reviewer.')
        ''', (
            rev_id, audio_id, pred_result["consistency_status"],
            pred_result["final_class"], pred_result["final_class"]
        ))

    conn.commit()
    conn.close()

    return jsonify({
        "success": True,
        "audio_id": audio_id,
        "filename": filename,
        "duration": round(duration, 2),
        "sha256": sha256_hash,
        "quality": quality_result,
        "prediction": pred_result,
        "alert": alert_result,
        "waveform_url": wave_url,
        "spectrogram_url": spec_url
    })


@app.route('/api/audio/live-chunk', methods=['POST'])
@login_required
def api_audio_live_chunk():
    """
    Ingests live 1.5s sliding window PCM chunks from client Web Audio API (SRS §10).
    Evaluates real-time threat detection and consecutive window confirmation.
    """
    data = request.get_data()
    if not data or len(data) < 100:
        return jsonify({"error": "Empty audio buffer"}), 400

    samples = parse_pcm_buffer(data)
    quality = assess_audio_quality(samples)
    features = extract_spectral_features(samples)
    prediction = classify_audio_dual(samples, features, quality, filename="live_mic_stream.raw")
    session_id = f"user_{session.get('user_id', 1)}"
    alert = evaluate_alert(prediction, session_id=session_id)

    return jsonify({
        "success": True,
        "quality": quality,
        "prediction": prediction,
        "alert": alert,
        "timestamp": datetime.utcnow().strftime("%H:%M:%S")
    })


@app.route('/api/audio/events', methods=['GET'])
@login_required
def api_audio_events():
    events = database.get_all_audio_events()
    return jsonify([dict(e) for e in events])


@app.route('/api/audio/<audio_id>', methods=['GET'])
@login_required
def api_audio_detail(audio_id):
    ev = database.get_audio_event_by_id(audio_id)
    if not ev:
        return jsonify({"error": "Audio event not found"}), 404

    d = dict(ev)
    if d.get('python_probabilities'):
        try:
            d['python_probabilities'] = json.loads(d['python_probabilities'])
        except Exception:
            pass
    if d.get('gtm_probabilities'):
        try:
            d['gtm_probabilities'] = json.loads(d['gtm_probabilities'])
        except Exception:
            pass
    return jsonify(d)


@app.route('/api/alerts/active', methods=['GET'])
@login_required
def api_active_alerts():
    alerts = database.get_active_alerts()
    return jsonify([dict(a) for a in alerts])


@app.route('/api/alerts/<alert_id>/ack', methods=['POST'])
@login_required
def api_alert_ack(alert_id):
    status = request.json.get('status', 'Acknowledged') if request.is_json else 'Acknowledged'
    database.update_alert_status(alert_id, status, user_id=session.get('user_id'))
    return jsonify({"success": True, "alert_id": alert_id, "new_status": status})


@app.route('/api/review/queue', methods=['GET'])
@login_required
def api_review_queue():
    queue = database.get_review_queue()
    return jsonify([dict(q) for q in queue])


@app.route('/api/review/<review_id>/override', methods=['POST'])
@login_required
@roles_required(database.ROLE_AUDIO_REVIEWER, database.ROLE_ADMINISTRATOR)
def api_review_override(review_id):
    payload = request.get_json() or request.form
    final_decision = payload.get('final_decision')
    comments = payload.get('comments', 'Forensic override applied.')
    if not final_decision:
        return jsonify({"error": "final_decision is required"}), 400

    success, msg = database.submit_review_override(
        review_id, final_decision, comments, reviewer_id=session.get('user_id')
    )
    if success:
        return jsonify({"success": True, "message": msg})
    return jsonify({"error": msg}), 400


@app.route('/api/reports/<audio_id>/pdf', methods=['GET'])
@login_required
def api_download_report_pdf(audio_id):
    ev = database.get_audio_event_by_id(audio_id)
    if not ev:
        abort(404, "Audio event not found")

    pdf_bytes = generate_incident_pdf(dict(ev))
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"SonicSentinel_Incident_{audio_id}.pdf"
    )


@app.route('/api/admin/rules/reload', methods=['POST'])
@login_required
@roles_required(database.ROLE_ADMINISTRATOR)
def api_reload_rules():
    success = reload_rules()
    return jsonify({"reloaded": success, "message": "Alert policies successfully hot-reloaded."})


if __name__ == '__main__':
    print("==========================================================")
    print("  SonicSentinel AI - NextWave Acoustic Intelligence")
    print("  Enterprise Acoustic Threat Detection & Monitoring System")
    print("  Server running on http://127.0.0.1:5000")
    print("  Default Logins:")
    print("    Admin:       admin       |  Password: Admin@123")
    print("    Security:    operator    |  Password: Operator@123")
    print("    Reviewer:    reviewer    |  Password: Reviewer@123")
    print("    Maintenance: maintenance |  Password: Maint@123")
    print("    User:        user        |  Password: User@123")
    print("==========================================================")
    app.run(host='127.0.0.1', port=5000, debug=True)


