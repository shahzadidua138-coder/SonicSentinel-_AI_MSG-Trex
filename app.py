"""
SonicSentinel AI - NextWave Acoustic Intelligence Web Application
Backend Server using Python Flask and SQLite Authentication
"""

import os
import csv
import io
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response, abort
)
import database

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
    """Export standard events to CSV file"""
    output = io.StringIO()
    writer = csv.writer(output)

    headers = [
        "Audio_ID", "Filename", "Category", "Python_Confidence",
        "GTM_Confidence", "Confidence_Delta", "Audio_Quality",
        "Severity", "Status", "Timestamp"
    ]
    writer.writerow(headers)

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
    print("  Default Logins:")
    print("    Admin:       admin       |  Password: Admin@123")
    print("    Security:    operator    |  Password: Operator@123")
    print("    Reviewer:    reviewer    |  Password: Reviewer@123")
    print("    Maintenance: maintenance |  Password: Maint@123")
    print("    User:        user        |  Password: User@123")
    print("==========================================================")
    app.run(host='127.0.0.1', port=5000, debug=True)

