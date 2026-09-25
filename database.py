"""
database.py - SQLite Database Management for SonicSentinel AI
Handles user credentials, roles, secure password hashing, and session queries.
Implements complete acoustic persistence: audio_records, predictions, alerts, reviews, audit_logs.
"""

import sqlite3
import os
import json
import uuid
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'sonicsentinel.db')

# Canonical system roles
ROLE_ADMINISTRATOR = 'Administrator'
ROLE_SECURITY_OPERATOR = 'Security Operator'
ROLE_AUDIO_REVIEWER = 'Audio Reviewer'
ROLE_MAINTENANCE_OPERATOR = 'Maintenance Operator'
ROLE_NORMAL_USER = 'Normal User'

ALL_ROLES = [
    ROLE_ADMINISTRATOR,
    ROLE_SECURITY_OPERATOR,
    ROLE_AUDIO_REVIEWER,
    ROLE_MAINTENANCE_OPERATOR,
    ROLE_NORMAL_USER
]

# Only these roles can be created by Administrator
ADMIN_CREATABLE_ROLES = [
    ROLE_SECURITY_OPERATOR,
    ROLE_AUDIO_REVIEWER,
    ROLE_MAINTENANCE_OPERATOR
]


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Enable WAL mode for high performance concurrency
    try:
        cursor.execute('PRAGMA journal_mode=WAL;')
    except Exception:
        pass

    # 1. Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'Normal User',
            organization TEXT DEFAULT 'Apex Acoustic Defense Operations',
            station TEXT DEFAULT 'Terminal #04 (Sector B)',
            theme_preference TEXT DEFAULT 'light',
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')

    # Migration: Ensure is_active column exists
    cursor.execute('PRAGMA table_info(users)')
    columns = [col[1] for col in cursor.fetchall()]
    if 'is_active' not in columns:
        cursor.execute('ALTER TABLE users ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1')

    # Migration: Normalize existing role names in database
    cursor.execute("UPDATE users SET role = 'Normal User' WHERE LOWER(role) IN ('normal user', 'normal_user')")
    cursor.execute("UPDATE users SET role = 'Maintenance Operator' WHERE LOWER(role) IN ('maintenance operator', 'maintenance_operator')")
    cursor.execute("UPDATE users SET role = 'Security Operator' WHERE LOWER(role) IN ('security operator', 'security_operator')")
    cursor.execute("UPDATE users SET role = 'Audio Reviewer' WHERE LOWER(role) IN ('audio reviewer', 'audio_reviewer')")
    cursor.execute("UPDATE users SET role = 'Administrator' WHERE LOWER(role) IN ('administrator', 'admin')")

    # 2. Audio Records table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audio_records (
            id TEXT PRIMARY KEY,
            user_id INTEGER,
            source_type TEXT NOT NULL,
            file_path TEXT,
            original_filename TEXT,
            duration_seconds REAL NOT NULL,
            sample_rate INTEGER NOT NULL DEFAULT 22050,
            channels INTEGER DEFAULT 1,
            sha256_hash TEXT,
            quality_grade TEXT NOT NULL DEFAULT 'Good',
            is_clipped INTEGER DEFAULT 0,
            is_silent INTEGER DEFAULT 0,
            snr_db REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # 3. Predictions table (Dual AI Models)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            audio_id TEXT UNIQUE NOT NULL,
            python_predicted_class TEXT NOT NULL,
            python_top_confidence REAL NOT NULL,
            python_probabilities TEXT NOT NULL,
            gtm_predicted_class TEXT NOT NULL,
            gtm_top_confidence REAL NOT NULL,
            gtm_probabilities TEXT NOT NULL,
            confidence_difference REAL NOT NULL,
            top_two_margin REAL NOT NULL,
            consistency_status TEXT NOT NULL,
            final_detected_class TEXT NOT NULL,
            waveform_image_path TEXT,
            spectrogram_image_path TEXT,
            evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (audio_id) REFERENCES audio_records(id)
        )
    ''')

    # 4. Alerts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id TEXT PRIMARY KEY,
            audio_id TEXT NOT NULL,
            threat_category TEXT NOT NULL,
            severity TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Active',
            is_confirmed_repeated INTEGER DEFAULT 0,
            recommended_action TEXT NOT NULL,
            acknowledged_by INTEGER,
            acknowledged_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (audio_id) REFERENCES audio_records(id),
            FOREIGN KEY (acknowledged_by) REFERENCES users(id)
        )
    ''')

    # 5. Reviews table (Triage Queue & Overrides)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id TEXT PRIMARY KEY,
            audio_id TEXT NOT NULL,
            reviewer_id INTEGER,
            triage_reason TEXT NOT NULL,
            original_decision TEXT NOT NULL,
            final_decision TEXT NOT NULL,
            is_override INTEGER DEFAULT 0,
            reviewer_comments TEXT NOT NULL,
            reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (audio_id) REFERENCES audio_records(id),
            FOREIGN KEY (reviewer_id) REFERENCES users(id)
        )
    ''')

    # 6. Audit Logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_logs (
            id TEXT PRIMARY KEY,
            user_id INTEGER,
            action TEXT NOT NULL,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Ensure built-in Administrator account exists
    cursor.execute("SELECT id FROM users WHERE LOWER(username) = 'admin' OR LOWER(role) = 'administrator'")
    admin_exists = cursor.fetchone()

    if not admin_exists:
        admin_pass_hash = generate_password_hash('Admin@123')
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, full_name, role, organization, station, theme_preference, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        ''', (
            'admin',
            'admin@sonicsentinel.ai',
            admin_pass_hash,
            'System Administrator',
            ROLE_ADMINISTRATOR,
            'Apex Acoustic Defense Command',
            'Command Central #01',
            'light'
        ))

    # Ensure default evaluation accounts exist
    cursor.execute("SELECT id FROM users WHERE LOWER(username) = 'operator'")
    if not cursor.fetchone():
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, full_name, role, organization, station, theme_preference, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        ''', ('operator', 'operator@sonicsentinel.ai', generate_password_hash('Operator@123'), 'Capt. Alexander Vance', ROLE_SECURITY_OPERATOR, 'Apex Acoustic Defense Operations', 'Terminal #04 (Sector B)', 'light'))

    cursor.execute("SELECT id FROM users WHERE LOWER(username) = 'reviewer'")
    if not cursor.fetchone():
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, full_name, role, organization, station, theme_preference, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        ''', ('reviewer', 'reviewer@sonicsentinel.ai', generate_password_hash('Reviewer@123'), 'Dr. Elena Rostova', ROLE_AUDIO_REVIEWER, 'Acoustic Forensics Lab', 'Lab Station #02', 'light'))

    cursor.execute("SELECT id FROM users WHERE LOWER(username) = 'maintenance'")
    if not cursor.fetchone():
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, full_name, role, organization, station, theme_preference, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        ''', ('maintenance', 'maintenance@sonicsentinel.ai', generate_password_hash('Maint@123'), 'Eng. Marcus Vance', ROLE_MAINTENANCE_OPERATOR, 'Facility Engineering Dept', 'HVAC Diagnostic Dock', 'light'))

    cursor.execute("SELECT id FROM users WHERE LOWER(username) = 'user'")
    if not cursor.fetchone():
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, full_name, role, organization, station, theme_preference, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        ''', ('user', 'user@sonicsentinel.ai', generate_password_hash('User@123'), 'Claire Thompson', ROLE_NORMAL_USER, 'General Security Division', 'Observation Post #03', 'light'))

    conn.commit()

    # Seed the 11 mandatory competition scenarios if not present
    seed_competition_scenarios(conn)

    conn.close()


def seed_competition_scenarios(conn):
    """
    Seeds the 11 Mandatory Competition Test Scenarios from SRS §13:
    AUD-01: Confirmed Gunshot
    AUD-02: Panic Scream
    AUD-03: Glass Breaking
    AUD-04: Machinery Fault
    AUD-05: Animal Sound
    AUD-06: Help Request
    AUD-07: Model Disagreement
    AUD-08: Severe Audio Clipping
    AUD-09: Silent Recording
    AUD-10: Duplicate Audio File
    AUD-11: Boundary Noise Case
    """
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM audio_records WHERE id LIKE 'AUD-%'")
    count = cursor.fetchone()[0]
    if count >= 11:
        return

    admin_user = cursor.execute("SELECT id FROM users WHERE role = 'Administrator' LIMIT 1").fetchone()
    admin_id = admin_user['id'] if admin_user else 1

    scenarios = [
        {
            "id": "AUD-01",
            "filename": "test_gunshot.wav",
            "duration": 0.8,
            "quality": "Good",
            "clipped": 0, "silent": 0, "snr": 31.5,
            "hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "py_class": "Gunshot", "py_conf": 0.92,
            "gtm_class": "Gunshot", "gtm_conf": 0.89,
            "delta": 0.03, "margin": 0.88, "status": "Strong Match",
            "final": "Gunshot", "severity": "Critical", "alert_status": "Active",
            "repeated": 1,
            "action": "CRITICAL LOCKDOWN: Dispatch armed security patrol to Sector East. Alert municipal emergency response."
        },
        {
            "id": "AUD-02",
            "filename": "test_scream.wav",
            "duration": 1.5,
            "quality": "Good",
            "clipped": 0, "silent": 0, "snr": 25.8,
            "hash": "a4f89d32b509f6e1e670a48b941569427e5a8f4c32b509f6e1e670a48b941569",
            "py_class": "Panic Scream", "py_conf": 0.88,
            "gtm_class": "Panic Scream", "gtm_conf": 0.85,
            "delta": 0.03, "margin": 0.79, "status": "Strong Match",
            "final": "Panic Scream", "severity": "Critical", "alert_status": "Escalated",
            "repeated": 1,
            "action": "EMERGENCY SAFETY ALERT: Dispatch on-site medical and rapid response unit to Stairwell B."
        },
        {
            "id": "AUD-03",
            "filename": "test_glass_break.wav",
            "duration": 1.2,
            "quality": "Good",
            "clipped": 0, "silent": 0, "snr": 27.2,
            "hash": "7c98b67f12e8731b54a20b9e8a01f68745c98e14620f8d9513a0c7e48b6154ef",
            "py_class": "Glass Breaking", "py_conf": 0.86,
            "gtm_class": "Glass Breaking", "gtm_conf": 0.83,
            "delta": 0.03, "margin": 0.76, "status": "Strong Match",
            "final": "Glass Breaking", "severity": "High", "alert_status": "Active",
            "repeated": 0,
            "action": "PERIMETER ALARM: Direct PTZ Camera #04 to North Storage Annex. Verify window barrier."
        },
        {
            "id": "AUD-04",
            "filename": "machinery_bearing_cavitation.wav",
            "duration": 3.0,
            "quality": "Good",
            "clipped": 0, "silent": 0, "snr": 22.4,
            "hash": "9b12c58e4a7d6f03e28c4b9175a30e8624f9a5d718e2c0b4395617df8a42e5c1",
            "py_class": "Machinery Fault", "py_conf": 0.84,
            "gtm_class": "Machinery Fault", "gtm_conf": 0.81,
            "delta": 0.03, "margin": 0.72, "status": "Strong Match",
            "final": "Machinery Fault", "severity": "High", "alert_status": "Acknowledged",
            "repeated": 1,
            "action": "MECHANICAL ADVISORY: Schedule bearing lubrication and harmonic vibration diagnostic for Pump #02."
        },
        {
            "id": "AUD-05",
            "filename": "test_dog_bark.mp3",
            "duration": 1.0,
            "quality": "Good",
            "clipped": 0, "silent": 0, "snr": 29.1,
            "hash": "3e84a29c15b70d4f68e91c2a57b04e3895c12f47a6b83d09e52714fa9c8b36e1",
            "py_class": "Animal Sound", "py_conf": 0.90,
            "gtm_class": "Animal Sound", "gtm_conf": 0.88,
            "delta": 0.02, "margin": 0.82, "status": "Strong Match",
            "final": "Animal Sound", "severity": "Low", "alert_status": "Closed",
            "repeated": 0,
            "action": "FACILITY AUDIT: Perimeter motion telemetry logged. Domestic canine acoustic profile confirmed."
        },
        {
            "id": "AUD-06",
            "filename": "help_request_emergency.wav",
            "duration": 2.4,
            "quality": "Acceptable",
            "clipped": 0, "silent": 0, "snr": 19.5,
            "hash": "5d29a74c10e38b6f91c5a72e48b30d9564e18f2a7c390b5d8416e2a9b74c53d8",
            "py_class": "Person Asking for Help", "py_conf": 0.87,
            "gtm_class": "Person Asking for Help", "gtm_conf": 0.84,
            "delta": 0.03, "margin": 0.78, "status": "Strong Match",
            "final": "Person Asking for Help", "severity": "Critical", "alert_status": "Active",
            "repeated": 1,
            "action": "SAFETY ASSISTANCE: Distress phrase 'Help me! Emergency!' recognized. Open two-way audio to Loading Bay."
        },
        {
            "id": "AUD-07",
            "filename": "metallic_clank_ambiguous.wav",
            "duration": 2.1,
            "quality": "Acceptable",
            "clipped": 0, "silent": 0, "snr": 16.3,
            "hash": "1c74b89e32a50d6f48e29c1b75a03d8495f21e8a6c470b9d5218e7a4b39c65e2",
            "py_class": "Machinery Fault", "py_conf": 0.72,
            "gtm_class": "Alarm or Siren", "gtm_conf": 0.68,
            "delta": 0.04, "margin": 0.12, "status": "Model Disagreement",
            "final": "Machinery Fault", "severity": "High", "alert_status": "Active",
            "repeated": 0,
            "action": "MANUAL REVIEW REQUIRED: Divergent model predictions. Queued for Audio Reviewer forensic arbitration."
        },
        {
            "id": "AUD-08",
            "filename": "test_clipping_audio.wav",
            "duration": 1.8,
            "quality": "Unusable",
            "clipped": 1, "silent": 0, "snr": 4.2,
            "hash": "8f31b79c24a60d5e17c49b2a86d05f9372e35a1c84b90d7e632914fa5b8c27e4",
            "py_class": "Gunshot", "py_conf": 0.58,
            "gtm_class": "Glass Breaking", "gtm_conf": 0.52,
            "delta": 0.06, "margin": 0.06, "status": "Unusable Quality",
            "final": "Quality Rejected", "severity": "Informational", "alert_status": "Closed",
            "repeated": 0,
            "action": "QUALITY GATE REJECTION: Audio clipping exceeds 1.0% limit. Automated threat triggering suspended."
        },
        {
            "id": "AUD-09",
            "filename": "test_silent_recording.wav",
            "duration": 3.0,
            "quality": "Unusable",
            "clipped": 0, "silent": 1, "snr": 0.0,
            "hash": "6b28a95d13e47c0a82b58d3f94c16e7250f14a8b73d92e5c410789fa6c5b14e3",
            "py_class": "Background Noise", "py_conf": 0.99,
            "gtm_class": "Background Noise", "gtm_conf": 0.99,
            "delta": 0.00, "margin": 0.98, "status": "Silent Recording",
            "final": "Background Noise", "severity": "Informational", "alert_status": "Closed",
            "repeated": 0,
            "action": "SILENCE SUPPRESSION: Signal energy RMS < 0.005. Recorded as clean ambient baseline."
        },
        {
            "id": "AUD-10",
            "filename": "duplicate_gunshot_sample.wav",
            "duration": 0.8,
            "quality": "Good",
            "clipped": 0, "silent": 0, "snr": 31.5,
            "hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "py_class": "Gunshot", "py_conf": 0.92,
            "gtm_class": "Gunshot", "gtm_conf": 0.89,
            "delta": 0.03, "margin": 0.88, "status": "Duplicate Audio Hash",
            "final": "Gunshot", "severity": "Critical", "alert_status": "Active",
            "repeated": 1,
            "action": "DUPLICATE HASH FLAGGED: Exact match to AUD-01. Linked to historical incident cluster."
        },
        {
            "id": "AUD-11",
            "filename": "rain_with_distant_horn.wav",
            "duration": 2.5,
            "quality": "Acceptable",
            "clipped": 0, "silent": 0, "snr": 12.8,
            "hash": "2f49c81a56e03b7d89f14e2c95a08d6371b48e3a9c520d7f416892fa3b7c85e9",
            "py_class": "Background Noise", "py_conf": 0.61,
            "gtm_class": "Vehicle Horn", "gtm_conf": 0.58,
            "delta": 0.03, "margin": 0.03, "status": "Uncertain Result",
            "final": "Background Noise", "severity": "Informational", "alert_status": "Active",
            "repeated": 0,
            "action": "UNCERTAIN RESULT: Top-two class margin < 0.10. Dispatched to triage queue for manual validation."
        }
    ]

    for s in scenarios:
        # Insert Audio Record
        cursor.execute('''
            INSERT OR REPLACE INTO audio_records (
                id, user_id, source_type, file_path, original_filename,
                duration_seconds, sample_rate, channels, sha256_hash,
                quality_grade, is_clipped, is_silent, snr_db
            ) VALUES (?, ?, 'demo', ?, ?, ?, 22050, 1, ?, ?, ?, ?, ?)
        ''', (
            s["id"], admin_id, f"sample_audio/{s['filename']}", s["filename"],
            s["duration"], s["hash"], s["quality"], s["clipped"], s["silent"], s["snr"]
        ))

        # Dummy probabilities for 10 classes
        py_probs = {s["py_class"]: s["py_conf"]}
        gtm_probs = {s["gtm_class"]: s["gtm_conf"]}

        # Insert Prediction
        cursor.execute('''
            INSERT OR REPLACE INTO predictions (
                id, audio_id, python_predicted_class, python_top_confidence,
                python_probabilities, gtm_predicted_class, gtm_top_confidence,
                gtm_probabilities, confidence_difference, top_two_margin,
                consistency_status, final_detected_class, waveform_image_path,
                spectrogram_image_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            f"PRED-{s['id']}", s["id"], s["py_class"], s["py_conf"],
            json.dumps(py_probs), s["gtm_class"], s["gtm_conf"],
            json.dumps(gtm_probs), s["delta"], s["margin"],
            s["status"], s["final"],
            f"/static/plots/{s['id']}_wave.png", f"/static/plots/{s['id']}_spec.png"
        ))

        # Insert Alert
        cursor.execute('''
            INSERT OR REPLACE INTO alerts (
                id, audio_id, threat_category, severity, status,
                is_confirmed_repeated, recommended_action
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            f"ALT-{s['id']}", s["id"], s["final"], s["severity"], s["alert_status"],
            s["repeated"], s["action"]
        ))

        # If model disagreement or uncertain, add to reviews queue
        if s["status"] in ["Model Disagreement", "Uncertain Result", "Unusable Quality"]:
            cursor.execute('''
                INSERT OR REPLACE INTO reviews (
                    id, audio_id, triage_reason, original_decision,
                    final_decision, is_override, reviewer_comments
                ) VALUES (?, ?, ?, ?, ?, 0, 'Pending forensic evaluation by Audio Reviewer.')
            ''', (
                f"REV-{s['id']}", s["id"], s["status"], s["final"], s["final"]
            ))

    conn.commit()


# ==========================================
# AUDIO & INCIDENT QUERY METHODS
# ==========================================

def get_all_audio_events():
    conn = get_db_connection()
    events = conn.execute('''
        SELECT a.id, a.original_filename, a.duration_seconds, a.quality_grade, a.snr_db, a.created_at,
               p.python_predicted_class, p.python_top_confidence,
               p.gtm_predicted_class, p.gtm_top_confidence,
               p.confidence_difference, p.top_two_margin, p.consistency_status, p.final_detected_class,
               alt.severity, alt.status as alert_status, alt.recommended_action
        FROM audio_records a
        LEFT JOIN predictions p ON a.id = p.audio_id
        LEFT JOIN alerts alt ON a.id = alt.audio_id
        ORDER BY a.created_at DESC
    ''').fetchall()
    conn.close()
    return events


def get_audio_event_by_id(audio_id):
    conn = get_db_connection()
    event = conn.execute('''
        SELECT a.*, p.python_predicted_class, p.python_top_confidence, p.python_probabilities,
               p.gtm_predicted_class, p.gtm_top_confidence, p.gtm_probabilities,
               p.confidence_difference, p.top_two_margin, p.consistency_status, p.final_detected_class,
               p.waveform_image_path, p.spectrogram_image_path,
               alt.id as alert_id, alt.severity, alt.status as alert_status,
               alt.recommended_action, alt.is_confirmed_repeated
        FROM audio_records a
        LEFT JOIN predictions p ON a.id = p.audio_id
        LEFT JOIN alerts alt ON a.id = alt.audio_id
        WHERE a.id = ?
    ''', (audio_id,)).fetchone()
    conn.close()
    return event


def get_active_alerts():
    conn = get_db_connection()
    alerts = conn.execute('''
        SELECT alt.*, a.original_filename, a.quality_grade, a.snr_db,
               p.python_predicted_class, p.python_top_confidence,
               p.gtm_predicted_class, p.gtm_top_confidence,
               p.confidence_difference, p.consistency_status
        FROM alerts alt
        JOIN audio_records a ON alt.audio_id = a.id
        LEFT JOIN predictions p ON a.id = p.audio_id
        ORDER BY 
            CASE alt.severity 
                WHEN 'Critical' THEN 1 
                WHEN 'High' THEN 2 
                WHEN 'Medium' THEN 3 
                ELSE 4 
            END,
            alt.created_at DESC
    ''').fetchall()
    conn.close()
    return alerts


def update_alert_status(alert_id, status, user_id=None):
    conn = get_db_connection()
    conn.execute('''
        UPDATE alerts 
        SET status = ?, acknowledged_by = ?, acknowledged_at = ?
        WHERE id = ?
    ''', (status, user_id, datetime.utcnow() if status != 'Active' else None, alert_id))
    conn.commit()
    conn.close()
    return True


def get_review_queue():
    conn = get_db_connection()
    reviews = conn.execute('''
        SELECT r.*, a.original_filename, a.duration_seconds, a.quality_grade, a.snr_db,
               p.python_predicted_class, p.python_top_confidence,
               p.gtm_predicted_class, p.gtm_top_confidence,
               p.confidence_difference, p.consistency_status
        FROM reviews r
        JOIN audio_records a ON r.audio_id = a.id
        LEFT JOIN predictions p ON a.id = p.audio_id
        ORDER BY r.reviewed_at DESC
    ''').fetchall()
    conn.close()
    return reviews


def submit_review_override(review_id, final_decision, comments, reviewer_id):
    conn = get_db_connection()
    # Check if overriding
    review = conn.execute('SELECT * FROM reviews WHERE id = ?', (review_id,)).fetchone()
    if not review:
        conn.close()
        return False, "Review item not found."

    is_override = 1 if final_decision != review['original_decision'] else 0
    conn.execute('''
        UPDATE reviews
        SET final_decision = ?, is_override = ?, reviewer_comments = ?, reviewer_id = ?, reviewed_at = ?
        WHERE id = ?
    ''', (final_decision, is_override, comments, reviewer_id, datetime.utcnow(), review_id))

    # Also update final_detected_class in prediction
    conn.execute('''
        UPDATE predictions
        SET final_detected_class = ?
        WHERE audio_id = ?
    ''', (final_decision, review['audio_id']))

    # Also update threat_category in alert
    conn.execute('''
        UPDATE alerts
        SET threat_category = ?, status = 'Reviewed'
        WHERE audio_id = ?
    ''', (final_decision, review['audio_id']))

    conn.commit()
    conn.close()
    return True, "Forensic override successfully recorded."


def log_audit_action(user_id, action, details=None):
    conn = get_db_connection()
    log_id = f"LOG-{uuid.uuid4().hex[:8].upper()}"
    conn.execute('''
        INSERT INTO audit_logs (id, user_id, action, details)
        VALUES (?, ?, ?, ?)
    ''', (log_id, user_id, action, json.dumps(details) if isinstance(details, dict) else str(details)))
    conn.commit()
    conn.close()
    return log_id


# ==========================================
# USER & RBAC METHODS (PRESERVED)
# ==========================================

def get_user_by_id(user_id):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    return user


def get_user_by_email_or_username(identifier):
    conn = get_db_connection()
    user = conn.execute(
        'SELECT * FROM users WHERE LOWER(email) = LOWER(?) OR LOWER(username) = LOWER(?)',
        (identifier, identifier)
    ).fetchone()
    conn.close()
    return user


def get_all_users():
    """Retrieve all users for Admin Operator Management"""
    conn = get_db_connection()
    users = conn.execute('''
        SELECT id, username, email, full_name, role, organization, station, is_active, created_at, last_login
        FROM users
        ORDER BY 
            CASE role 
                WHEN 'Administrator' THEN 1 
                WHEN 'Security Operator' THEN 2 
                WHEN 'Audio Reviewer' THEN 3 
                WHEN 'Maintenance Operator' THEN 4 
                ELSE 5 
            END,
            created_at DESC
    ''').fetchall()
    conn.close()
    return users


def create_user(username, email, password, full_name, role=ROLE_NORMAL_USER, organization='Apex Acoustic Defense Operations', station='Terminal #04 (Sector B)', is_active=1):
    """
    Creates a new user record.
    Enforces password hashing and checks username/email uniqueness.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check duplicate username
    cursor.execute('SELECT id FROM users WHERE LOWER(username) = LOWER(?)', (username,))
    if cursor.fetchone():
        conn.close()
        return False, "Username is already registered. Please choose another."

    # Check duplicate email
    cursor.execute('SELECT id FROM users WHERE LOWER(email) = LOWER(?)', (email,))
    if cursor.fetchone():
        conn.close()
        return False, "Email address is already registered. Please sign in or use another email."

    password_hash = generate_password_hash(password)

    cursor.execute('''
        INSERT INTO users (username, email, password_hash, full_name, role, organization, station, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (username, email, password_hash, full_name, role, organization, station, is_active))

    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return True, user_id


def toggle_user_status(user_id):
    """Activates or deactivates an account. Admin cannot be deactivated."""
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    if not user:
        conn.close()
        return False, "User not found."

    if user['role'] == ROLE_ADMINISTRATOR:
        conn.close()
        return False, "The built-in Administrator account cannot be deactivated."

    new_status = 0 if user['is_active'] == 1 else 1
    conn.execute('UPDATE users SET is_active = ? WHERE id = ?', (new_status, user_id))
    conn.commit()
    conn.close()
    status_label = "activated" if new_status == 1 else "deactivated"
    return True, f"User account '{user['username']}' has been {status_label}."


def verify_user_credentials(identifier, password):
    """
    Verifies user credentials.
    Checks password hash and enforces is_active account status.
    """
    user = get_user_by_email_or_username(identifier)
    if not user:
        return None, "Invalid credentials. Please verify your username/email and password."

    if not check_password_hash(user['password_hash'], password):
        return None, "Invalid credentials. Please verify your username/email and password."

    # Check active status
    if user['is_active'] == 0:
        return None, "Your account has been deactivated. Please contact an administrator."

    # Update last login timestamp
    conn = get_db_connection()
    conn.execute('UPDATE users SET last_login = ? WHERE id = ?', (datetime.utcnow(), user['id']))
    conn.commit()
    conn.close()

    return user, None


def update_user_profile(user_id, full_name, station, theme_preference):
    conn = get_db_connection()
    conn.execute('''
        UPDATE users
        SET full_name = ?, station = ?, theme_preference = ?
        WHERE id = ?
    ''', (full_name, station, theme_preference, user_id))
    conn.commit()
    conn.close()
    return True


def reset_user_password(identifier, new_password):
    """Securely updates password for a verified user by email or username"""
    user = get_user_by_email_or_username(identifier)
    if not user:
        return False, "No account associated with this username or email was found."

    if user['is_active'] == 0:
        return False, "Account is deactivated. Password reset is not permitted."

    new_hash = generate_password_hash(new_password)
    conn = get_db_connection()
    conn.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_hash, user['id']))
    conn.commit()
    conn.close()
    return True, "Your password has been successfully reset. You may now log in."
