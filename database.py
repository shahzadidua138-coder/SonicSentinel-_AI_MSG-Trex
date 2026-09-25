"""
database.py - SQLite Database Management for SonicSentinel AI
Handles user credentials, roles, secure password hashing, and session queries.
"""

import sqlite3
import os
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

    # Users table
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
    conn.close()


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


