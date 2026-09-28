"""
SonicSentinel AI - NextWave Acoustic Intelligence Web Application
Backend Server using Python Flask and Neon PostgreSQL Authentication
Simplified Auth: Only 'admin' and 'user' roles.
Users get website. Admin gets dashboard.
"""

import os
import csv
import io
import json
import uuid
import wave
import numpy as np
from datetime import datetime
from functools import wraps
from werkzeug.utils import secure_filename
from flask import (
    Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response, abort, send_file, send_from_directory
)
import database
from src.utils.audio_hash import compute_audio_hash
from src.services.quality_assessor import assess_audio_quality
from src.services.feature_extractor import extract_spectral_features
from src.services.audio_dsp import (
    load_audio_file, parse_pcm_buffer, render_waveform_image, render_spectrogram_image
)
from src.services.ml_service import classify_audio_dual, get_model_runtime_status
from src.services.alert_engine import evaluate_alert, reload_rules
from src.services.pdf_generator import generate_incident_pdf
from config.settings import UPLOAD_FOLDER, CATEGORIES as MANDATORY_CLASS_LABELS

app = Flask(__name__)
# Cryptographically signed sessions
app.secret_key = os.environ.get('SECRET_KEY', 'sonicsentinel-secure-session-key-2026-acousticx')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Initialize the PostgreSQL schema. Demo data is opt-in via SEED_DEMO_DATA=true.
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


@app.errorhandler(413)
def request_too_large(error):
    if request.path.startswith('/api/'):
        return jsonify({"error": "Request exceeds the 50 MB upload limit."}), 413
    return "Upload exceeds the 50 MB limit.", 413


# ==========================================
# ASSET SERVING ROUTE
# ==========================================

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve static asset files directly from assets folder"""
    assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
    return send_from_directory(assets_dir, filename)


# ==========================================
# AUTHENTICATION DECORATORS
# ==========================================

def login_required(f):
    """
    Ensures route is accessible only to active, authenticated sessions.
    Validates user against database on every request to prevent deactivated account access.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            if request.path.startswith('/api/'):
                return jsonify({"error": "Authentication required. Please sign in and try again."}), 401
            flash("Please sign in to access this page.", "warning")
            return redirect(url_for('login', next=request.url))

        # Query database to confirm user exists and is active
        user = database.get_user_by_id(session['user_id'])
        if not user:
            session.clear()
            if request.path.startswith('/api/'):
                return jsonify({"error": "Your session is invalid. Please sign in again."}), 401
            flash("Your session is invalid. Please sign in again.", "error")
            return redirect(url_for('login'))

        if user['is_active'] == 0:
            session.clear()
            if request.path.startswith('/api/'):
                return jsonify({"error": "Your account is inactive."}), 403
            flash("Your account has been deactivated or suspended. Please contact an administrator.", "error")
            return redirect(url_for('login'))

        # Synchronize session state with database
        session['role'] = user['role']
        session['full_name'] = user['full_name']
        session['username'] = user['username']

        return f(*args, **kwargs)
    return decorated_function


def role_required(allowed_roles):
    """
    RBAC decorator: restricts access to specific roles.
    If authenticated user does not have permission, aborts with HTTP 403 Forbidden.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                flash("Please sign in to access this page.", "warning")
                return redirect(url_for('login', next=request.url))

            user = database.get_user_by_id(session['user_id'])
            if not user or user['is_active'] == 0:
                session.clear()
                flash("Your account has been deactivated or suspended. Please contact an administrator.", "error")
                return redirect(url_for('login'))

            user_role = database.normalize_role(user['role'])
            session['role'] = user_role
            session['full_name'] = user['full_name']
            session['username'] = user['username']

            norm_allowed = [database.normalize_role(r) for r in allowed_roles]
            if user_role not in norm_allowed:
                abort(403)

            return f(*args, **kwargs)
        return decorated_function
    return decorator


def admin_required(f):
    """Ensures route is accessible only to Administrator. Aborts with 403 Forbidden otherwise."""
    return role_required([database.ROLE_ADMIN])(f)


@app.errorhandler(403)
def forbidden_error(e):
    """Renders custom 403 Forbidden template with HTTP 403 status code."""
    if request.path.startswith('/api/'):
        return jsonify({"error": "Your account does not have permission to perform this action."}), 403
    return render_template(
        '403.html',
        user_role=session.get('role', 'Normal User'),
        target_endpoint=request.path
    ), 403



@app.context_processor
def inject_user_and_context():
    """Inject current user information and metadata into all templates"""
    current_user = None
    if 'user_id' in session:
        current_user = database.get_user_by_id(session['user_id'])

    return {
        'current_user': current_user,
        'app_name': 'SonicSentinel AI',
        'system_version': '1.0.0',
        'system_uptime': '99.94%',
        'now': datetime.utcnow  # provides now() in templates
    }



# ==========================================
# PUBLIC AUTHENTICATION ROUTES
# ==========================================

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('website_home'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    # If user is already logged in, redirect directly to their dashboard
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

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
        session['role'] = database.normalize_role(user['role'])
        session['full_name'] = user['full_name']
        session.permanent = remember

        flash(f"Welcome back, {user['full_name']}!", "success")

        # Determine target: check next_page parameter
        next_page = request.args.get('next')
        user_role = database.normalize_role(user['role'])
        if next_page and not any(auth_path in next_page for auth_path in ['/login', '/register', '/logout']):
            if user_role == database.ROLE_USER and any(p in next_page for p in ['/profile', '/dashboard', 'operator', 'reviewer', 'admin', 'maintenance']):
                return redirect(url_for('website_home'))
            return redirect(next_page)

        # Redirect to role-specific dashboard (Normal User goes to website_home)
        return redirect(url_for('dashboard'))

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    # If user is already logged in, redirect to dashboard
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        full_name = request.form.get('full_name', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()

        # SECURITY RULE: Public register users ALWAYS become 'Normal User'
        assigned_role = database.ROLE_USER

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
            station='Community Safety Portal',
            is_admin_provision=False
        )

        if not success:
            flash(result, "error")
            return render_template('register.html', username=username, email=email, full_name=full_name)

        flash("Registration successful! Your account is ready. Please sign in.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')



@app.route('/forgot-password', methods=['GET', 'POST'])
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if 'user_id' in session:
        return redirect(url_for('website_home'))

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
    flash("You have been securely signed out.", "info")
    return redirect(url_for('login'))


# ==========================================
# WEBSITE ROUTES (FOR ALL USERS)
# ==========================================

@app.route('/home')
def website_home():
    return render_template('website_home.html')


@app.route('/assets/<path:filename>')
def serve_asset(filename):
    """Serve user-provided visual assets such as the active brand mark."""
    return send_from_directory(os.path.join(app.root_path, 'assets'), filename)


@app.route('/assets/brand-logo')
def brand_logo():
    """Serve the user's logo.jpg when present, with the current PNG as fallback."""
    assets_dir = os.path.join(app.root_path, 'assets')
    filename = 'logo.jpg' if os.path.exists(os.path.join(assets_dir, 'logo.jpg')) else 'logo.png'
    return send_from_directory(assets_dir, filename)


@app.route('/about')
def website_about():
    return render_template('website_about.html')


@app.route('/features')
def website_features():
    return render_template('website_features.html')


@app.route('/technology')
def website_technology():
    return render_template('website_technology.html')


@app.route('/contact', methods=['GET', 'POST'])
def website_contact():
    """Contact page with simple form handling"""
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        if not name or not email or not message:
            flash('All fields are required.', 'error')
            return render_template('website_contact.html')
        flash(f'Thank you, {name}! Your message has been received.', 'success')
        return redirect(url_for('website_home'))
    return render_template('website_contact.html')


# ==========================================
# ROLE DASHBOARD ROUTES (SRS SECTION 1.6)
# ==========================================

@app.route('/dashboard')
@login_required
def dashboard():
    """
    Universal Dashboard Router:
    Dispatches authenticated user to their role-specific dashboard.
    """
    role = database.normalize_role(session.get('role', database.ROLE_USER))
    if role == database.ROLE_ADMIN:
        return redirect(url_for('admin_dashboard'))
    elif role == database.ROLE_SECURITY:
        return redirect(url_for('operator_dashboard'))
    elif role == database.ROLE_REVIEWER:
        return redirect(url_for('reviewer_dashboard'))
    elif role == database.ROLE_MAINTENANCE:
        return redirect(url_for('maintenance_dashboard'))
    else:
        return redirect(url_for('website_home'))


@app.route('/user/dashboard')
@login_required
def user_dashboard():
    """
    Normal User / Community Safety Portal (SRS §1.6 lxiii)
    Normal users are routed to the public website; operators/admin can preview.
    """
    role = database.normalize_role(session.get('role', database.ROLE_USER))
    if role == database.ROLE_USER:
        return redirect(url_for('website_home'))

    user_id = session.get('user_id')
    user = database.get_user_by_id(user_id)
    events = database.get_all_audio_events()
    user_events = [e for e in events if e['user_id'] == user_id] or events[:6]
    stats = database.get_system_overview_stats()
    return render_template(
        'user_dashboard.html',
        user=user,
        events=user_events,
        stats=stats,
        active_page='user_dashboard'
    )


@app.route('/operator/dashboard')
@login_required
@role_required([database.ROLE_SECURITY, database.ROLE_ADMIN])
def operator_dashboard():
    """
    Security Operator Incident Command Center (SRS §1.6 lxiv)
    Accessible to Security Operator and Administrator.
    """
    user = database.get_user_by_id(session['user_id'])
    alerts = database.get_active_alerts()
    events = database.get_all_audio_events()
    stats = database.get_system_overview_stats()
    return render_template(
        'operator_dashboard.html',
        user=user,
        alerts=alerts,
        events=events[:10],
        stats=stats,
        active_page='operator_dashboard'
    )


@app.route('/reviewer/dashboard')
@login_required
@role_required([database.ROLE_REVIEWER, database.ROLE_ADMIN])
def reviewer_dashboard():
    """
    Audio Reviewer Forensic Laboratory (SRS §1.6 lvii-lxi)
    Accessible to Audio Reviewer and Administrator.
    """
    user = database.get_user_by_id(session['user_id'])
    queue = database.get_review_queue()
    events = database.get_all_audio_events()
    stats = database.get_system_overview_stats()
    return render_template(
        'reviewer_dashboard.html',
        user=user,
        queue=queue,
        events=events,
        stats=stats,
        active_page='reviewer_dashboard'
    )


@app.route('/maintenance/dashboard')
@login_required
@role_required([database.ROLE_MAINTENANCE, database.ROLE_ADMIN])
def maintenance_dashboard():
    """
    Maintenance Operator Diagnostic Dock (SRS §1.6 vii, xli)
    Accessible to Maintenance Operator and Administrator.
    """
    user = database.get_user_by_id(session['user_id'])
    events = database.get_all_audio_events()
    stats = database.get_system_overview_stats()
    return render_template(
        'maintenance_dashboard.html',
        user=user,
        events=events,
        stats=stats,
        active_page='maintenance_dashboard'
    )


@app.route('/admin/dashboard')
@login_required
@admin_required
def admin_dashboard():
    """
    Administrator Control Center & Personnel Governance (SRS §1.6 lxv)
    Restricted to Administrator only.
    """
    users = database.get_all_users()
    events = database.get_all_audio_events()
    alerts = database.get_active_alerts()
    stats = database.get_system_overview_stats()
    role_counts = database.get_user_role_counts()
    cat_dist = database.get_category_distribution()
    return render_template(
        'admin_dashboard.html',
        users=users,
        events=events,
        alerts=alerts,
        stats=stats,
        role_counts=role_counts,
        cat_dist=cat_dist,
        assignable_roles=database.ALL_ASSIGNABLE_ROLES,
        active_page='admin_dashboard'
    )


# ==========================================
# ADMIN USER MANAGEMENT CRUD ENDPOINTS
# ==========================================

@app.route('/admin/create-operator', methods=['POST'])
@login_required
@admin_required
def admin_create_operator():
    """
    Allows Administrator to provision accounts with any of the 4 assignable roles.
    (Kept for compatibility with test suites & admin form)
    """
    full_name = request.form.get('full_name', '').strip()
    username = request.form.get('username', '').strip()
    email = request.form.get('email', '').strip()
    role = request.form.get('role', database.ROLE_SECURITY).strip()
    station = request.form.get('station', '').strip()
    organization = request.form.get('organization', 'SonicSentinel Operations').strip()
    password = request.form.get('password', '').strip()

    if not full_name or not username or not email or not password:
        flash("All fields are required to create an account.", "error")
        return redirect(url_for('admin_dashboard'))

    norm_role = database.normalize_role(role)
    if norm_role not in database.ALL_ASSIGNABLE_ROLES:
        flash("Security Policy: Administrator accounts can only be provisioned directly in the database.", "error")
        return redirect(url_for('admin_dashboard'))

    success, result = database.create_user(
        username=username,
        email=email,
        password=password,
        full_name=full_name,
        role=norm_role,
        organization=organization,
        station=station,
        is_admin_provision=True
    )
    if success:
        database.log_audit_action(session.get('user_id'), "CREATE_USER", f"Created {norm_role} account for {username}")
        flash(f"Account for {full_name} (@{username}) successfully created with role '{norm_role}'.", "success")
    else:
        flash(result, "error")
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/users/create', methods=['POST'])
@login_required
@admin_required
def admin_users_create():
    """Admin CRUD: Create user endpoint (aliases create_operator)"""
    return admin_create_operator()


@app.route('/admin/users/<int:user_id>/edit', methods=['POST'])
@login_required
@admin_required
def admin_users_edit(user_id):
    """
    Admin CRUD: Edit existing user details, station, password, or role.
    Target role is restricted to the 4 operational roles (cannot escalate to Admin).
    """
    full_name = request.form.get('full_name', '').strip()
    email = request.form.get('email', '').strip()
    role = request.form.get('role', '').strip()
    station = request.form.get('station', '').strip()
    organization = request.form.get('organization', '').strip()
    password = request.form.get('password', '').strip()

    if not full_name or not email:
        flash("Full name and email address are required.", "error")
        return redirect(url_for('admin_dashboard'))

    success, msg = database.update_user(
        user_id=user_id,
        full_name=full_name,
        email=email,
        role=role if role else None,
        station=station,
        organization=organization,
        password=password if password else None
    )
    if success:
        database.log_audit_action(session.get('user_id'), "EDIT_USER", f"Updated details for user ID {user_id}")
        flash(msg, "success")
    else:
        flash(msg, "error")
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/users/<int:user_id>/change-role', methods=['POST'])
@login_required
@admin_required
def admin_users_change_role(user_id):
    """
    Admin CRUD: One-click change user's role to any of the 4 assignable roles:
    - Normal User
    - Security Operator
    - Audio Reviewer
    - Maintenance Operator
    (Administrator cannot be assigned via UI, only DB).
    """
    target_role = request.form.get('role') or (request.json.get('role') if request.is_json else None)
    if not target_role:
        if request.is_json:
            return jsonify({"success": False, "error": "Target role is required"}), 400
        flash("Target role is required.", "error")
        return redirect(url_for('admin_dashboard'))

    success, msg = database.update_user_role(user_id, target_role)
    if success:
        database.log_audit_action(session.get('user_id'), "ROLE_CHANGE", f"Changed user {user_id} role to {target_role}")
        if request.is_json:
            return jsonify({"success": True, "message": msg})
        flash(msg, "success")
    else:
        if request.is_json:
            return jsonify({"success": False, "error": msg}), 400
        flash(msg, "error")
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@login_required
@admin_required
def admin_users_delete(user_id):
    """Admin CRUD: Permanently delete a user account (built-in admin and self protected)"""
    if user_id == session.get('user_id'):
        flash("Access Denied: You cannot delete your currently active administrator account.", "error")
        return redirect(url_for('admin_dashboard'))

    success, msg = database.delete_user(user_id)
    if success:
        database.log_audit_action(session.get('user_id'), "DELETE_USER", f"Deleted user ID {user_id}")
        if request.is_json:
            return jsonify({"success": True, "message": msg})
        flash(msg, "success")
    else:
        if request.is_json:
            return jsonify({"success": False, "error": msg}), 400
        flash(msg, "error")
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/toggle-user/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def admin_toggle_user(user_id):
    """
    Admin CRUD: Suspend or activate a user account.
    Suspended accounts are locked out immediately.
    """
    if user_id == session.get('user_id'):
        flash("Access Denied: You cannot suspend your currently active administrator account.", "error")
        return redirect(url_for('admin_dashboard'))

    success, message = database.toggle_user_status(user_id)
    if success:
        database.log_audit_action(session.get('user_id'), "TOGGLE_USER_STATUS", f"Toggled status for user {user_id}")
        if request.is_json:
            return jsonify({"success": True, "message": message})
        flash(message, "success")
    else:
        if request.is_json:
            return jsonify({"success": False, "error": message}), 400
        flash(message, "error")
    return redirect(url_for('admin_dashboard'))


@app.route('/api/admin/users', methods=['GET'])
@login_required
@admin_required
def api_admin_users():
    """Returns JSON listing of all registered users and role counts for interactive UI"""
    users = database.get_all_users()
    counts = database.get_user_role_counts()
    return jsonify({
        "success": True,
        "users": [dict(u) for u in users],
        "role_counts": counts
    })



# ==========================================
# PROTECTED MODULE ROUTES (AUTH REQUIRED)
# ==========================================

@app.route('/audio-upload')
@login_required
def audio_upload():
    """Audio upload is accessible to all authenticated users"""
    return render_template('audio_upload.html', active_page='audio_upload')


@app.route('/audio-analysis')
@login_required
def audio_analysis():
    return render_template('audio_analysis.html', active_page='audio_analysis')


@app.route('/live-monitoring')
@login_required
def live_monitoring():
    return render_template('live_monitoring.html', active_page='live_monitoring')


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
    """Event history is accessible to all authenticated users"""
    return render_template('event_history.html', active_page='events')


@app.route('/analytics')
@login_required
def analytics():
    return render_template('analytics.html', active_page='analytics')


@app.route('/reports')
@login_required
def reports():
    events = database.get_all_audio_events()
    user = database.get_user_by_id(session.get('user_id'))
    if user and database.normalize_role(user['role']) == database.ROLE_USER:
        events = [event for event in events if event['user_id'] == session.get('user_id')]
    return render_template('reports.html', events=events, active_page='reports')


@app.route('/settings')
@login_required
@admin_required
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
@role_required([database.ROLE_ADMIN])
def export_csv():
    """Export standard events from SQLite database to CSV file"""
    output = io.StringIO()
    writer = csv.writer(output)

    headers = [
        "Audio_ID", "Filename", "Category", "Python_Confidence",
        "Random_Forest_Confidence", "Confidence_Delta", "Audio_Quality",
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
    extension = os.path.splitext(filename)[1].lower()
    if extension not in {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}:
        return jsonify({"error": "Unsupported audio format. Upload WAV, MP3, FLAC, OGG, or M4A."}), 415
    audio_bytes = file.read()
    if len(audio_bytes) == 0:
        return jsonify({"error": "Uploaded file is 0 bytes"}), 400
    if len(audio_bytes) > 50 * 1024 * 1024:
        return jsonify({"error": "Audio file exceeds the 50 MB upload limit."}), 413

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

    # 2. Capture source metadata before DSP standardizes audio to mono/22.05 kHz.
    try:
        import soundfile as sf
        try:
            source_info = sf.info(saved_path)
            source_format, source_rate, source_channels, source_subtype = source_info.format, source_info.samplerate, source_info.channels, source_info.subtype
        except Exception:
            import librosa
            source_data, source_rate = librosa.load(saved_path, sr=None, mono=False)
            source_channels = 1 if source_data.ndim == 1 else source_data.shape[0]
            source_format, source_subtype = extension.lstrip('.').upper(), None
        source_metadata = {
            "format": source_format,
            "sample_rate": source_rate,
            "channels": source_channels,
            "subtype": source_subtype,
            "bit_depth": int(source_subtype.rsplit('_', 1)[-1]) if source_subtype and source_subtype.rsplit('_', 1)[-1].isdigit() else None,
            "file_size_bytes": len(audio_bytes),
        }
        samples, sr, duration = load_audio_file(saved_path)
    except Exception as exc:
        try:
            os.remove(saved_path)
        except OSError:
            pass
        return jsonify({"error": f"Audio could not be decoded: {exc}"}), 422
    if duration < 0.5 or duration > 30:
        os.remove(saved_path)
        return jsonify({"error": "Audio duration must be between 0.5 and 30 seconds.", "duration": round(duration, 3)}), 422

    # 3. Audio Quality Gatekeeper
    quality_result = assess_audio_quality(samples, sr)

    # 4. Feature Extraction
    features = extract_spectral_features(samples, sr)

    # 5. Dual AI Model Inference & Arbitration
    try:
        pred_result = classify_audio_dual(samples, features, quality_result, filename=filename)
    except Exception as exc:
        os.remove(saved_path)
        app.logger.exception("Audio model inference failed")
        return jsonify({"error": f"Audio analysis failed because a trained model or feature extractor is unavailable: {exc}"}), 503

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
            duration_seconds, audio_format, sample_rate, channels, bit_depth, file_size_bytes, sha256_hash,
            quality_grade, is_clipped, is_silent, snr_db
        ) VALUES (?, ?, 'upload', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        audio_id, user_id, saved_path, filename,
        round(duration, 2), source_metadata['format'], source_metadata['sample_rate'],
        source_metadata['channels'], source_metadata['bit_depth'], source_metadata['file_size_bytes'], sha256_hash,
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
            ) VALUES (?, ?, ?, ?, ?, 0, 'Pending forensic evaluation.')
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
        "metadata": {**source_metadata, "duration_seconds": round(duration, 3), "upload_time": datetime.utcnow().isoformat() + 'Z'},
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
    if len(samples) < 22050 or len(samples) > 66000:
        return jsonify({"error": "Live audio windows must contain between 1 and 3 seconds of audio."}), 422
    quality = assess_audio_quality(samples)
    features = extract_spectral_features(samples)
    try:
        prediction = classify_audio_dual(samples, features, quality, filename="live_mic_stream.raw")
    except Exception as exc:
        app.logger.exception("Live audio model inference failed")
        return jsonify({"error": f"Live audio analysis is unavailable: {exc}"}), 503
    session_id = f"user_{session.get('user_id', 1)}"
    alert = evaluate_alert(prediction, session_id=session_id)

    # Persist only actionable or uncertain windows; normal ambient microphone
    # audio stays in memory and is never silently recorded.
    live_audio_id = None
    if alert.get('alert_triggered'):
        try:
            pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(22050)
                wav_file.writeframes(pcm16.tobytes())
            wav_bytes = wav_buffer.getvalue()
            live_audio_id = f"AUD-{uuid.uuid4().hex[:8].upper()}"
            live_path = os.path.join(UPLOAD_FOLDER, f"{live_audio_id}_live-window.wav")
            with open(live_path, 'wb') as stored_audio:
                stored_audio.write(wav_bytes)
            file_hash = compute_audio_hash(wav_bytes)
            wave_url = render_waveform_image(samples, live_audio_id)
            spec_url = render_spectrogram_image(samples, live_audio_id)
            conn = database.get_db_connection()
            conn.execute('''INSERT INTO audio_records
                (id, user_id, source_type, file_path, original_filename, duration_seconds,
                 sample_rate, channels, sha256_hash, quality_grade, is_clipped, is_silent, snr_db)
                VALUES (?, ?, 'live', ?, 'live-window.wav', ?, 22050, 1, ?, ?, ?, ?, ?)''',
                (live_audio_id, session.get('user_id'), live_path, len(samples) / 22050,
                 file_hash, quality['quality_grade'], int(quality['is_clipped']),
                 int(quality['is_silent']), quality['snr_db']))
            conn.execute('''INSERT INTO predictions
                (id, audio_id, python_predicted_class, python_top_confidence, python_probabilities,
                 gtm_predicted_class, gtm_top_confidence, gtm_probabilities,
                 confidence_difference, top_two_margin, consistency_status, final_detected_class,
                 waveform_image_path, spectrogram_image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
                (f"PRED-{uuid.uuid4().hex[:8].upper()}", live_audio_id,
                 prediction['python_class'], prediction['python_confidence'], json.dumps(prediction['python_probabilities']),
                 prediction['gtm_class'], prediction['gtm_confidence'], json.dumps(prediction['gtm_probabilities']),
                 prediction['confidence_difference'], prediction['top_two_margin'], prediction['consistency_status'],
                 prediction['final_class'], wave_url, spec_url))
            conn.execute('''INSERT INTO alerts
                (id, audio_id, threat_category, severity, status, is_confirmed_repeated, recommended_action)
                VALUES (?, ?, ?, ?, ?, ?, ?)''',
                (f"ALT-{uuid.uuid4().hex[:8].upper()}", live_audio_id, alert['threat_category'], alert['severity'],
                 alert['status'], int(alert['is_confirmed_repeated']), alert['recommended_action']))
            if prediction['consistency_status'] in ('Model Disagreement', 'Uncertain Result'):
                conn.execute('''INSERT INTO reviews
                    (id, audio_id, triage_reason, original_decision, final_decision, is_override, reviewer_comments)
                    VALUES (?, ?, ?, ?, ?, 0, ?)''',
                    (f"REV-{uuid.uuid4().hex[:8].upper()}", live_audio_id, prediction['consistency_status'],
                     prediction['final_class'], prediction['final_class'], 'Pending live-window review.'))
            conn.commit()
            conn.close()
            database.log_audit_action(session.get('user_id'), 'LIVE_EVENT_CAPTURED', live_audio_id)
        except Exception:
            app.logger.exception("Could not persist actionable live audio window")

    return jsonify({
        "success": True,
        "quality": quality,
        "prediction": prediction,
        "alert": alert,
        "audio_id": live_audio_id,
        "timestamp": datetime.utcnow().strftime("%H:%M:%S")
    })


@app.route('/api/audio/events', methods=['GET'])
@login_required
def api_audio_events():
    events = database.get_all_audio_events()
    user = database.get_user_by_id(session.get('user_id'))
    if user and database.normalize_role(user['role']) == database.ROLE_USER:
        events = [event for event in events if event['user_id'] == session.get('user_id')]
    return jsonify([dict(e) for e in events])


@app.route('/api/audio/<audio_id>', methods=['GET'])
@login_required
def api_audio_detail(audio_id):
    ev = database.get_audio_event_by_id(audio_id)
    if not ev:
        return jsonify({"error": "Audio event not found"}), 404
    user = database.get_user_by_id(session.get('user_id'))
    if user and database.normalize_role(user['role']) == database.ROLE_USER and ev['user_id'] != session.get('user_id'):
        abort(403)

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
    d['comparison_model'] = {
        "name": "Random Forest",
        "predicted_class": d.get('gtm_predicted_class'),
        "confidence": d.get('gtm_top_confidence'),
        "probabilities": d.get('gtm_probabilities'),
    }
    d.pop('file_path', None)
    return jsonify(d)


@app.route('/api/audio/<audio_id>/file', methods=['GET'])
@login_required
def api_audio_file(audio_id):
    """Serve the original recording only to its owner or an operational role."""
    ev = database.get_audio_event_by_id(audio_id)
    if not ev:
        abort(404, "Audio event not found")
    user = database.get_user_by_id(session.get('user_id'))
    role = database.normalize_role(user['role']) if user else database.ROLE_USER
    if role == database.ROLE_USER and ev['user_id'] != session.get('user_id'):
        abort(403)
    audio_path = ev['file_path']
    if not audio_path or not os.path.isfile(audio_path):
        abort(404, "Stored audio is unavailable")
    return send_file(audio_path, as_attachment=False, download_name=ev['original_filename'] or f"{audio_id}.wav")


@app.route('/api/alerts/active', methods=['GET'])
@login_required
def api_active_alerts():
    alerts_list = database.get_active_alerts()
    user = database.get_user_by_id(session.get('user_id'))
    if user and database.normalize_role(user['role']) == database.ROLE_USER:
        owned_ids = {event['id'] for event in database.get_all_audio_events() if event['user_id'] == session.get('user_id')}
        alerts_list = [alert for alert in alerts_list if alert['audio_id'] in owned_ids]
    return jsonify([dict(a) for a in alerts_list])


@app.route('/api/alerts/<alert_id>/ack', methods=['POST'])
@login_required
@role_required([database.ROLE_SECURITY, database.ROLE_ADMIN])
def api_alert_ack(alert_id):
    status = request.json.get('status', 'Acknowledged') if request.is_json else 'Acknowledged'
    if status not in {'Acknowledged', 'Dismissed', 'Escalated'}:
        return jsonify({"error": "Status must be Acknowledged, Dismissed, or Escalated."}), 400
    if not database.update_alert_status(alert_id, status, user_id=session.get('user_id')):
        return jsonify({"error": "Alert not found."}), 404
    return jsonify({"success": True, "alert_id": alert_id, "new_status": status})


@app.route('/api/review/queue', methods=['GET'])
@login_required
@role_required([database.ROLE_REVIEWER, database.ROLE_ADMIN])
def api_review_queue():
    queue = database.get_review_queue()
    return jsonify([dict(q) for q in queue])


@app.route('/api/review/<review_id>/override', methods=['POST'])
@login_required
@role_required([database.ROLE_REVIEWER, database.ROLE_ADMIN])
def api_review_override(review_id):
    payload = request.get_json() or request.form
    final_decision = payload.get('final_decision')
    comments = str(payload.get('comments', 'Override applied.')).strip()[:2000]
    if not final_decision:
        return jsonify({"error": "final_decision is required"}), 400
    if final_decision == 'Aggression or Violent Conflict':
        final_decision = 'Aggression'
    allowed_decisions = set(MANDATORY_CLASS_LABELS) | {'Unknown', 'Unknown Sound'}
    if final_decision not in allowed_decisions:
        return jsonify({"error": "Decision must be one of the supported sound classes or Unknown."}), 400

    success, msg = database.submit_review_override(
        review_id, final_decision, comments, reviewer_id=session.get('user_id')
    )
    if success:
        database.log_audit_action(session.get('user_id'), 'REVIEW_OVERRIDE', {
            'review_id': review_id, 'decision': final_decision, 'comments': comments
        })
        return jsonify({"success": True, "message": msg})
    return jsonify({"error": msg}), 400


@app.route('/api/reports/<audio_id>/pdf', methods=['GET'])
@login_required
def api_download_report_pdf(audio_id):
    ev = database.get_audio_event_by_id(audio_id)
    if not ev:
        abort(404, "Audio event not found")
    user = database.get_user_by_id(session.get('user_id'))
    if user and database.normalize_role(user['role']) == database.ROLE_USER and ev['user_id'] != session.get('user_id'):
        abort(403)

    pdf_bytes = generate_incident_pdf(dict(ev))
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"SonicSentinel_Incident_{audio_id}.pdf"
    )


@app.route('/api/admin/rules/reload', methods=['POST'])
@login_required
@admin_required
def api_reload_rules():
    success = reload_rules()
    return jsonify({"reloaded": success, "message": "Alert policies successfully hot-reloaded."})


@app.route('/Dashbords_bg_video/<path:filename>')
def serve_dashboard_bg_video(filename):
    dir_path = os.path.join(os.path.dirname(__file__), 'Dashbords_bg_video')
    return send_from_directory(dir_path, filename)


# ==========================================
# FAST REAL MODEL PREDICTION & SAMPLE APIS
# ==========================================

CATEGORY_FOLDER_MAP = {
    'gunshot': 'gunshot',
    'glass': 'Glass breaking',
    'machinery': 'machinery fault',
    'alarm': 'siren',
    'scream': 'scream',
    'noise': 'background noise',
    'aggression': 'aggression',
    'animal': 'animal_sound',
    'horn': 'vehicle_horn',
    'help': 'person_asking_for_help'
}

DATA_BASE_DIR = os.path.join(os.path.dirname(__file__), 'Model', 'data')


@app.route('/api/audio/sample/<category_key>', methods=['GET'])
def api_get_audio_sample(category_key):
    """Streams a real acoustic WAV/MP3 sample from the 3,000 clip dataset"""
    folder = CATEGORY_FOLDER_MAP.get(category_key.lower())
    if not folder:
        return jsonify({"error": f"Unknown category key: {category_key}"}), 404

    target_dir = os.path.join(DATA_BASE_DIR, folder)
    if not os.path.exists(target_dir):
        return jsonify({"error": "Dataset folder not found"}), 404

    files = [f for f in os.listdir(target_dir) if f.endswith(('.wav', '.mp3'))]
    if not files:
        return jsonify({"error": "No audio files available in category"}), 404

    sample_file = files[0]
    sample_path = os.path.join(target_dir, sample_file)
    mimetype = 'audio/wav' if sample_file.endswith('.wav') else 'audio/mpeg'
    return send_file(sample_path, mimetype=mimetype)


@app.route('/api/audio/classify-sample/<category_key>', methods=['GET', 'POST'])
def api_classify_sample(category_key):
    """
    Fast API: Runs real-time 373-feature extraction and trained dual-AI model inference
    on a real dataset clip, returning instant predictions to the frontend.
    """
    folder = CATEGORY_FOLDER_MAP.get(category_key.lower())
    if not folder:
        return jsonify({"error": f"Unknown category key: {category_key}"}), 404

    target_dir = os.path.join(DATA_BASE_DIR, folder)
    if not os.path.exists(target_dir):
        return jsonify({"error": "Dataset directory not found"}), 404

    files = [f for f in os.listdir(target_dir) if f.endswith(('.wav', '.mp3'))]
    if not files:
        return jsonify({"error": "No files found"}), 404

    sample_file = files[0]
    sample_path = os.path.join(target_dir, sample_file)

    try:
        samples, sr, _ = load_audio_file(sample_path)
    except Exception as e:
        app.logger.exception("Sample audio decoding failed")
        return jsonify({"error": f"Unable to decode sample audio: {e}"}), 422

    quality = assess_audio_quality(samples, sr)
    prediction = classify_audio_dual(samples, quality_result=quality, filename=sample_file)
    alert = evaluate_alert(prediction, session_id="sample_test")

    return jsonify({
        "success": True,
        "category_key": category_key,
        "sample_filename": sample_file,
        "sample_url": f"/api/audio/sample/{category_key}",
        "quality": quality,
        "prediction": prediction,
        "alert": alert
    })


@app.route('/api/model/metrics', methods=['GET'])
def api_model_metrics():
    """Returns the trained model evaluation metrics and confusion matrices"""
    metrics_path = os.path.join(os.path.dirname(__file__), 'Model', 'SonicSentinel', 'python_models', 'model_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        data['inference_models'] = {
            'primary': 'Support Vector Machine',
            'comparison': 'Random Forest',
            'google_teachable_machine_available': False,
        }
        data['runtime_status'] = get_model_runtime_status()
        return jsonify(data)
    return jsonify({"error": "Metrics file not found"}), 404



if __name__ == '__main__':
    print("==========================================================")
    print("  SonicSentinel AI - NextWave Acoustic Intelligence")
    print("  Server running on http://127.0.0.1:5000")
    print("  ")
    print("  Admin Login:  admin  |  Password: Admin@123")
    print("  Users: Register at /register")
    print("==========================================================")
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', '5000')), debug=True)
