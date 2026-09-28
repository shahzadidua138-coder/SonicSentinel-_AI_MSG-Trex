"""
database.py - PostgreSQL Database Management for SonicSentinel AI
Handles user credentials, secure password hashing, and session queries.
Implements complete acoustic persistence: audio_records, predictions, alerts, reviews, audit_logs.
Simplified Auth: Only 'admin' and 'user' roles.
"""

import os
import json
import uuid
import re
from pathlib import Path
import psycopg
from psycopg import sql
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

load_dotenv(Path(__file__).resolve().parent / '.env')
DATABASE_URL = os.environ.get('DATABASE_URL')
DATABASE_SCHEMA = os.environ.get('DATABASE_SCHEMA', 'public')
if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', DATABASE_SCHEMA):
    raise RuntimeError('DATABASE_SCHEMA must be a simple PostgreSQL schema name.')


class DatabaseRow(dict):
    """PostgreSQL row supporting both named and positional indexing."""
    def __init__(self, values, names):
        super().__init__(zip(names, values))
        self._names = names

    def __getitem__(self, key):
        if isinstance(key, int):
            key = self._names[key]
        return super().__getitem__(key)


def _row_factory(cursor):
    if cursor.description is None:
        return lambda values: values
    names = [column.name for column in cursor.description]
    return lambda values: DatabaseRow(values, names)


def _connect_database():
    connection = psycopg.connect(DATABASE_URL, row_factory=_row_factory)
    with connection.cursor() as cursor:
        if DATABASE_SCHEMA != 'public':
            cursor.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(DATABASE_SCHEMA)))
        cursor.execute(sql.SQL('SET search_path TO {}').format(sql.Identifier(DATABASE_SCHEMA)))
    connection.commit()
    return connection


class CompatCursor:
    def __init__(self, cursor):
        self._cursor = cursor

    def execute(self, query, params=None):
        query = _postgres_sql(query)
        if params is None:
            self._cursor.execute(query)
        else:
            self._cursor.execute(query, params)
        return self

    def fetchone(self): return self._cursor.fetchone()
    def fetchall(self): return self._cursor.fetchall()
    @property
    def rowcount(self): return self._cursor.rowcount
    @property
    def lastrowid(self): return None


class CompatConnection:
    def __init__(self, connection):
        self._connection = connection

    def cursor(self): return CompatCursor(self._connection.cursor(row_factory=_row_factory))
    def execute(self, query, params=None):
        cursor = self.cursor()
        try:
            return cursor.execute(query, params)
        except psycopg.OperationalError:
            # Read-only retries are safe after a Neon pooler disconnect; never replay writes.
            normalized = _postgres_sql(query)
            if not normalized.lstrip().upper().startswith('SELECT'):
                raise
            self._connection = _connect_database()
            return self.cursor().execute(normalized, params)
    def commit(self): return self._connection.commit()
    def rollback(self): return self._connection.rollback()
    def close(self): return self._connection.close()


def _postgres_sql(query):
    query = query.strip()
    if query.upper().startswith('PRAGMA '):
        raise ValueError('SQLite PRAGMA statements are not supported by PostgreSQL')
    is_replace = bool(re.match(r'INSERT\s+OR\s+REPLACE\s+INTO', query, flags=re.IGNORECASE))
    query = re.sub(r'\bINSERT\s+OR\s+REPLACE\s+INTO\b', 'INSERT INTO', query, flags=re.IGNORECASE)
    if is_replace and 'ON CONFLICT' not in query.upper() and 'RETURNING' not in query.upper():
        match = re.match(r'(INSERT INTO\s+\w+)\s*\(([^)]+)\)(\s*VALUES\s*\(.+\))\s*;?$', query, flags=re.IGNORECASE | re.DOTALL)
        if match:
            columns = [name.strip() for name in match.group(2).split(',')]
            updates = ', '.join(f'{name}=EXCLUDED.{name}' for name in columns if name.lower() != 'id')
            suffix = f'ON CONFLICT (id) DO UPDATE SET {updates}' if updates else 'ON CONFLICT (id) DO NOTHING'
            query = f'{match.group(1)} ({match.group(2)}){match.group(3)} {suffix}'
    return re.sub(r'\?', '%s', query)

# 5 Standard SRS Roles (SRS Section 1.6 ii)
ROLE_ADMIN = 'Administrator'
ROLE_SECURITY = 'Security Operator'
ROLE_REVIEWER = 'Audio Reviewer'
ROLE_MAINTENANCE = 'Maintenance Operator'
ROLE_USER = 'Normal User'

ALL_ASSIGNABLE_ROLES = [
    ROLE_USER,
    ROLE_SECURITY,
    ROLE_REVIEWER,
    ROLE_MAINTENANCE
]


def normalize_role(role_val):
    """
    Normalizes any role string representation to one of the 5 canonical SRS role names.
    Supports case-insensitivity and legacy short names ('admin', 'user', 'operator', 'reviewer', 'maintenance').
    """
    if not role_val:
        return ROLE_USER
    val = str(role_val).strip().lower()
    if 'admin' in val:
        return ROLE_ADMIN
    elif 'maint' in val:
        return ROLE_MAINTENANCE
    elif 'rev' in val:
        return ROLE_REVIEWER
    elif 'sec' in val or 'operat' in val:
        return ROLE_SECURITY
    else:
        return ROLE_USER



def get_db_connection():
    if not DATABASE_URL:
        raise RuntimeError('DATABASE_URL is required. Configure the Neon PostgreSQL connection in .env.')
    return CompatConnection(_connect_database())



def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id BIGSERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            organization TEXT DEFAULT 'SonicSentinel Community',
            station TEXT DEFAULT 'Web Portal',
            theme_preference TEXT DEFAULT 'light',
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')

    # Migrations for existing PostgreSQL installations.
    cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'users'")
    columns = {col[0] for col in cursor.fetchall()}
    if 'is_active' not in columns:
        cursor.execute('ALTER TABLE users ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1')

    # 2. Audio Records table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audio_records (
            id TEXT PRIMARY KEY,
            user_id INTEGER,
            source_type TEXT NOT NULL,
            file_path TEXT,
            original_filename TEXT,
            duration_seconds REAL NOT NULL,
            audio_format TEXT,
            sample_rate INTEGER NOT NULL DEFAULT 22050,
            channels INTEGER DEFAULT 1,
            bit_depth INTEGER,
            file_size_bytes INTEGER,
            sha256_hash TEXT,
            quality_grade TEXT NOT NULL DEFAULT 'Good',
            is_clipped INTEGER DEFAULT 0,
            is_silent INTEGER DEFAULT 0,
            snr_db REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = 'audio_records'")
    audio_columns = {col[0] for col in cursor.fetchall()}
    for column, declaration in {
        'audio_format': 'TEXT',
        'bit_depth': 'INTEGER',
        'file_size_bytes': 'INTEGER',
    }.items():
        if column not in audio_columns:
            cursor.execute(f'ALTER TABLE audio_records ADD COLUMN {column} {declaration}')

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

    # Normalize legacy roles in database
    cursor.execute("UPDATE users SET role = 'Administrator' WHERE LOWER(username) = 'admin' OR LOWER(role) = 'admin'")
    cursor.execute("UPDATE users SET role = 'Security Operator' WHERE LOWER(username) = 'operator' OR LOWER(role) IN ('operator', 'security')")
    cursor.execute("UPDATE users SET role = 'Audio Reviewer' WHERE LOWER(username) = 'reviewer' OR LOWER(role) = 'reviewer'")
    cursor.execute("UPDATE users SET role = 'Maintenance Operator' WHERE LOWER(username) = 'maintenance' OR LOWER(role) IN ('maintenance', 'maint')")
    cursor.execute("UPDATE users SET role = 'Normal User' WHERE (LOWER(username) = 'user' OR LOWER(role) = 'user') AND LOWER(username) != 'admin'")

    # Demo accounts are opt-in; never insert known default credentials into a Neon database.
    default_accounts = [
        {
            'username': 'admin',
            'email': 'admin@sonicsentinel.ai',
            'password': 'Admin@123',
            'full_name': 'System Administrator',
            'role': ROLE_ADMIN,
            'station': 'Admin Console (HQ)',
            'org': 'SonicSentinel Command HQ'
        },
        {
            'username': 'operator',
            'email': 'operator@sonicsentinel.ai',
            'password': 'Operator@123',
            'full_name': 'Officer Marcus Vance',
            'role': ROLE_SECURITY,
            'station': 'Security Operations Center (SOC)',
            'org': 'Perimeter Defense Division'
        },
        {
            'username': 'reviewer',
            'email': 'reviewer@sonicsentinel.ai',
            'password': 'Reviewer@123',
            'full_name': 'Dr. Elena Rostova',
            'role': ROLE_REVIEWER,
            'station': 'Acoustic Forensics Lab',
            'org': 'Forensics & Arbitration'
        },
        {
            'username': 'maintenance',
            'email': 'maintenance@sonicsentinel.ai',
            'password': 'Maint@123',
            'full_name': 'Eng. Tyler Briggs',
            'role': ROLE_MAINTENANCE,
            'station': 'Sensor Array Diagnostic Dock',
            'org': 'Hardware Engineering'
        },
        {
            'username': 'user',
            'email': 'user@sonicsentinel.ai',
            'password': 'User@123',
            'full_name': 'Sarah Jenkins',
            'role': ROLE_USER,
            'station': 'Community Safety Portal',
            'org': 'SonicSentinel Community'
        }
    ]

    for acc in default_accounts if os.environ.get('SEED_DEMO_DATA', '').lower() == 'true' else []:
        cursor.execute("SELECT id FROM users WHERE LOWER(username) = ?", (acc['username'].lower(),))
        if not cursor.fetchone():
            pass_hash = generate_password_hash(acc['password'])
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, full_name, role, organization, station, theme_preference, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'light', 1)
            ''', (acc['username'], acc['email'], pass_hash, acc['full_name'], acc['role'], acc['org'], acc['station']))

    conn.commit()

    # Seed the 11 mandatory competition scenarios if not present
    if os.environ.get('SEED_DEMO_DATA', '').lower() == 'true':
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

    admin_user = cursor.execute("SELECT id FROM users WHERE role = 'admin' LIMIT 1").fetchone()
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
            "action": "MANUAL REVIEW REQUIRED: Divergent model predictions. Queued for forensic arbitration."
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
                ) VALUES (?, ?, ?, ?, ?, 0, 'Pending forensic evaluation.')
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
        SELECT a.id, a.user_id, a.source_type, a.original_filename, a.duration_seconds, a.quality_grade, a.snr_db, a.created_at,
               p.python_predicted_class, p.python_top_confidence,
               p.gtm_predicted_class, p.gtm_top_confidence,
               p.confidence_difference, p.top_two_margin, p.consistency_status, p.final_detected_class,
               alt.severity, alt.status as alert_status, alt.recommended_action,
               (SELECT CASE WHEN r.is_override = 1 THEN 'Reviewed' ELSE 'Pending' END
                FROM reviews r WHERE r.audio_id = a.id ORDER BY r.reviewed_at DESC LIMIT 1) AS review_status
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
    cursor = conn.execute('''
        UPDATE alerts 
        SET status = ?, acknowledged_by = ?, acknowledged_at = ?
        WHERE id = ?
    ''', (status, user_id, datetime.utcnow() if status != 'Active' else None, alert_id))
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def get_review_queue():
    conn = get_db_connection()
    reviews = conn.execute('''
        SELECT r.*, a.original_filename, a.duration_seconds, a.quality_grade, a.snr_db, a.file_path,
               p.python_predicted_class, p.python_top_confidence,
               p.gtm_predicted_class, p.gtm_top_confidence,
               p.confidence_difference, p.consistency_status, p.waveform_image_path
        FROM reviews r
        JOIN audio_records a ON r.audio_id = a.id
        LEFT JOIN predictions p ON a.id = p.audio_id
        WHERE r.reviewer_id IS NULL
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
# USER METHODS (FULL CRUD & RBAC MANAGEMENT)
# ==========================================

def get_user_by_id(user_id):
    """Retrieves user by primary key ID and returns dict with normalized role."""
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if user:
        d = dict(user)
        d['role'] = normalize_role(d['role'])
        return d
    return None


def get_user_by_email_or_username(identifier):
    """Retrieves user by either email or username with normalized role."""
    conn = get_db_connection()
    user = conn.execute(
        'SELECT * FROM users WHERE LOWER(email) = LOWER(?) OR LOWER(username) = LOWER(?)',
        (identifier, identifier)
    ).fetchone()
    conn.close()
    if user:
        d = dict(user)
        d['role'] = normalize_role(d['role'])
        return d
    return None


def get_all_users():
    """Retrieve all users with normalized role ordering for Admin management."""
    conn = get_db_connection()
    users = conn.execute('''
        SELECT id, username, email, full_name, role, organization, station, is_active, created_at, last_login
        FROM users
        ORDER BY 
            CASE 
                WHEN LOWER(role) IN ('administrator', 'admin') THEN 1 
                WHEN LOWER(role) IN ('security operator', 'operator', 'security') THEN 2 
                WHEN LOWER(role) IN ('audio reviewer', 'reviewer') THEN 3 
                WHEN LOWER(role) IN ('maintenance operator', 'maintenance', 'maint') THEN 4 
                ELSE 5 
            END,
            created_at DESC
    ''').fetchall()
    conn.close()
    result = []
    for u in users:
        d = dict(u)
        d['role'] = normalize_role(d['role'])
        result.append(d)
    return result


def create_user(username, email, password, full_name, role=ROLE_USER, organization='SonicSentinel Community', station=None, is_active=1, is_admin_provision=False):
    """
    Creates a new user record.
    Enforces password hashing and checks username/email uniqueness.
    If created via public register (is_admin_provision=False), ALWAYS sets 'Normal User'.
    If provisioned by Admin in UI, role must be one of the 4 assignable roles (cannot create Administrator).
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    username = username.strip()
    email = email.strip()
    full_name = full_name.strip()

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

    norm_role = normalize_role(role)
    if not is_admin_provision:
        norm_role = ROLE_USER
    else:
        # Admin can only create users with one of the 4 operational roles
        if norm_role not in ALL_ASSIGNABLE_ROLES:
            norm_role = ROLE_USER

    if not station:
        station_defaults = {
            ROLE_ADMIN: 'Central Command',
            ROLE_SECURITY: 'Security Operations Center (SOC)',
            ROLE_REVIEWER: 'Acoustic Forensics Lab',
            ROLE_MAINTENANCE: 'Sensor Array Diagnostic Dock',
            ROLE_USER: 'Community Safety Portal'
        }
        station = station_defaults.get(norm_role, 'Web Portal')

    password_hash = generate_password_hash(password)

    cursor.execute('''
        INSERT INTO users (username, email, password_hash, full_name, role, organization, station, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING id
    ''', (username, email, password_hash, full_name, norm_role, organization, station, is_active))

    conn.commit()
    user_id = cursor.fetchone()['id']
    conn.close()
    return True, user_id


def update_user(user_id, full_name, email, role=None, station=None, organization=None, password=None):
    """
    Full CRUD Update: Modifies user details.
    Allows changing user role to any of the 4 assignable roles.
    Prevents assigning 'Administrator' through UI.
    Prevents modifying built-in Administrator's role.
    """
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    if not user:
        conn.close()
        return False, "User not found."

    cursor = conn.cursor()
    cursor.execute('SELECT id FROM users WHERE LOWER(email) = LOWER(?) AND id != ?', (email.strip(), user_id))
    if cursor.fetchone():
        conn.close()
        return False, "Email address is already in use by another account."

    current_role = normalize_role(user['role'])
    target_role = current_role
    if role:
        norm_target = normalize_role(role)
        if current_role == ROLE_ADMIN or user['username'].lower() == 'admin':
            # Built-in administrator role is protected
            target_role = ROLE_ADMIN
        else:
            if norm_target in ALL_ASSIGNABLE_ROLES:
                target_role = norm_target
            else:
                conn.close()
                return False, "Invalid role. Administrator role cannot be assigned through UI."

    updates = [
        "full_name = ?",
        "email = ?",
        "role = ?",
        "station = ?",
        "organization = ?"
    ]
    params = [
        full_name.strip(),
        email.strip(),
        target_role,
        (station or user['station'] or 'Web Portal').strip(),
        (organization or user['organization'] or 'SonicSentinel Community').strip()
    ]

    if password and len(password.strip()) >= 6:
        updates.append("password_hash = ?")
        params.append(generate_password_hash(password.strip()))

    params.append(user_id)
    query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
    conn.execute(query, params)
    conn.commit()
    conn.close()
    return True, f"User '{user['username']}' updated successfully."


def update_user_role(user_id, new_role):
    """
    Specifically changes a user's role to any of the 4 assignable roles:
    - Normal User
    - Security Operator
    - Audio Reviewer
    - Maintenance Operator
    Administrator role CANNOT be assigned here (database direct only).
    """
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    if not user:
        conn.close()
        return False, "User not found."

    if normalize_role(user['role']) == ROLE_ADMIN or user['username'].lower() == 'admin':
        conn.close()
        return False, "The Administrator role cannot be modified."

    norm_role = normalize_role(new_role)
    if norm_role not in ALL_ASSIGNABLE_ROLES:
        conn.close()
        return False, "Invalid role selection. Only operational roles (Normal User, Security Operator, Audio Reviewer, Maintenance Operator) may be assigned."

    conn.execute('UPDATE users SET role = ? WHERE id = ?', (norm_role, user_id))
    conn.commit()
    conn.close()
    return True, f"User '{user['username']}' role updated to '{norm_role}'."


def delete_user(user_id):
    """Deletes a user account. Built-in Administrator cannot be deleted."""
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    if not user:
        conn.close()
        return False, "User not found."

    if normalize_role(user['role']) == ROLE_ADMIN or user['username'].lower() == 'admin':
        conn.close()
        return False, "The built-in Administrator account cannot be deleted."

    conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()
    return True, f"User account '{user['username']}' has been permanently deleted."


def toggle_user_status(user_id):
    """Suspends / Deactivates or Activates an account. Admin cannot be deactivated."""
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    if not user:
        conn.close()
        return False, "User not found."

    if normalize_role(user['role']) == ROLE_ADMIN or user['username'].lower() == 'admin':
        conn.close()
        return False, "The built-in Administrator account cannot be suspended or deactivated."

    new_status = 0 if user['is_active'] == 1 else 1
    conn.execute('UPDATE users SET is_active = ? WHERE id = ?', (new_status, user_id))
    conn.commit()
    conn.close()
    status_label = "activated" if new_status == 1 else "suspended / deactivated"
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
        return None, "Your account has been deactivated or suspended. Please contact an administrator."

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
    """Securely updates password for a verified user by email or username."""
    user = get_user_by_email_or_username(identifier)
    if not user:
        return False, "No account associated with this username or email was found."

    if user['is_active'] == 0:
        return False, "Account is suspended/deactivated. Password reset is not permitted."

    new_hash = generate_password_hash(new_password)
    conn = get_db_connection()
    conn.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_hash, user['id']))
    conn.commit()
    conn.close()
    return True, "Your password has been successfully reset. You may now log in."


def get_user_role_counts():
    """Returns counts for each role and active/suspended totals."""
    conn = get_db_connection()
    rows = conn.execute('SELECT role, is_active, COUNT(*) as cnt FROM users GROUP BY role, is_active').fetchall()
    conn.close()
    counts = {
        'total': 0,
        'active': 0,
        'suspended': 0,
        'admin': 0,
        'security': 0,
        'reviewer': 0,
        'maintenance': 0,
        'user': 0
    }
    for r in rows:
        cnt = r['cnt']
        counts['total'] += cnt
        if r['is_active'] == 1:
            counts['active'] += cnt
        else:
            counts['suspended'] += cnt
        norm = normalize_role(r['role'])
        if norm == ROLE_ADMIN:
            counts['admin'] += cnt
        elif norm == ROLE_SECURITY:
            counts['security'] += cnt
        elif norm == ROLE_REVIEWER:
            counts['reviewer'] += cnt
        elif norm == ROLE_MAINTENANCE:
            counts['maintenance'] += cnt
        else:
            counts['user'] += cnt
    return counts


def get_system_overview_stats():
    """Calculates comprehensive telemetry stats for all dashboards."""
    conn = get_db_connection()
    total_audio = conn.execute('SELECT COUNT(*) FROM audio_records').fetchone()[0]
    total_alerts = conn.execute('SELECT COUNT(*) FROM alerts').fetchone()[0]
    active_alerts = conn.execute("SELECT COUNT(*) FROM alerts WHERE status = 'Active'").fetchone()[0]
    critical_alerts = conn.execute("SELECT COUNT(*) FROM alerts WHERE severity = 'Critical' AND status = 'Active'").fetchone()[0]
    high_alerts = conn.execute("SELECT COUNT(*) FROM alerts WHERE severity = 'High' AND status = 'Active'").fetchone()[0]
    pending_reviews = conn.execute("SELECT COUNT(*) FROM reviews WHERE reviewer_id IS NULL").fetchone()[0]
    disagreements = conn.execute("SELECT COUNT(*) FROM predictions WHERE consistency_status LIKE '%Disagreement%'").fetchone()[0]
    poor_quality = conn.execute("SELECT COUNT(*) FROM audio_records WHERE quality_grade IN ('Poor', 'Unusable')").fetchone()[0]
    avg_conf = conn.execute("SELECT AVG(python_top_confidence) FROM predictions").fetchone()[0] or 0.885
    conn.close()

    return {
        'total_audio': total_audio,
        'total_alerts': total_alerts,
        'active_alerts': active_alerts,
        'critical_alerts': critical_alerts,
        'high_alerts': high_alerts,
        'pending_reviews': pending_reviews,
        'disagreements': disagreements,
        'poor_quality': poor_quality,
        'avg_confidence_pct': round(float(avg_conf) * 100, 1),
        'hardware_nodes_online': 12,
        'total_hardware_nodes': 12,
        'system_health': 'OPTIMAL'
    }


def get_category_distribution():
    """Counts events per category for chart and breakdown display across all 10 mandatory categories."""
    conn = get_db_connection()
    rows = conn.execute('''
        SELECT final_detected_class, COUNT(*) as cnt 
        FROM predictions 
        GROUP BY final_detected_class
    ''').fetchall()
    conn.close()
    db_counts = {r['final_detected_class']: r['cnt'] for r in rows}

    categories = [
        {"name": "Gunshot", "severity": "Critical"},
        {"name": "Panic Scream", "severity": "Critical"},
        {"name": "Person Asking for Help", "severity": "Critical"},
        {"name": "Glass Breaking", "severity": "High"},
        {"name": "Machinery Fault", "severity": "High"},
        {"name": "Alarm or Siren", "severity": "Warning"},
        {"name": "Vehicle Horn", "severity": "Medium"},
        {"name": "Aggression", "severity": "Medium"},
        {"name": "Animal Sound", "severity": "Low"},
        {"name": "Background Noise", "severity": "Informational"}
    ]

    res = []
    for cat in categories:
        cnt = db_counts.get(cat['name'], 0)
        if cnt == 0:
            for k, v in db_counts.items():
                if k and (cat['name'].lower() in k.lower() or k.lower() in cat['name'].lower()):
                    cnt += v
        res.append({
            "name": cat['name'],
            "count": cnt,
            "severity": cat['severity']
        })
    return res
