"""
database.py - SQLite Database Management for SonicSentinel AI
Handles user credentials, roles, secure password hashing, and session queries.
"""

import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'sonicsentinel.db')


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
            role TEXT NOT NULL DEFAULT 'Security Operator',
            organization TEXT DEFAULT 'Apex Acoustic Defense Operations',
            station TEXT DEFAULT 'Terminal #04 (Sector B)',
            theme_preference TEXT DEFAULT 'dark',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')

    # Seed initial test enterprise users if database is empty
    cursor.execute('SELECT COUNT(*) FROM users')
    count = cursor.fetchone()[0]

    if count == 0:
        default_users = [
            (
                'admin',
                'admin@sonicsentinel.ai',
                generate_password_hash('Admin@123'),
                'Dr. Sarah Chen',
                'Administrator',
                'Apex Acoustic Defense Operations',
                'Command Central #01',
                'dark'
            ),
            (
                'operator',
                'operator@sonicsentinel.ai',
                generate_password_hash('Operator@123'),
                'Capt. Alexander Vance',
                'Security Operator',
                'Apex Acoustic Defense Operations',
                'Terminal #04 (Sector B)',
                'dark'
            ),
            (
                'reviewer',
                'reviewer@sonicsentinel.ai',
                generate_password_hash('Reviewer@123'),
                'Dr. Elena Rostova',
                'Audio Reviewer',
                'Acoustic Forensics Lab',
                'Lab Station #02',
                'light'
            ),
            (
                'maintenance',
                'maintenance@sonicsentinel.ai',
                generate_password_hash('Maint@123'),
                'Eng. Marcus Vance',
                'Maintenance operator',
                'Facility Engineering Dept',
                'HVAC Diagnostic Dock',
                'light'
            ),
            (
                'user',
                'user@sonicsentinel.ai',
                generate_password_hash('User@123'),
                'Claire Thompson',
                'Normal user',
                'General Security Division',
                'Observation Post #03',
                'light'
            )
        ]

        cursor.executemany('''
            INSERT INTO users (username, email, password_hash, full_name, role, organization, station, theme_preference)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', default_users)
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


def create_user(username, email, password, full_name, role='Security Operator', organization='Apex Acoustic Defense Operations', station='Terminal #04 (Sector B)'):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check duplicate username
    cursor.execute('SELECT id FROM users WHERE LOWER(username) = LOWER(?)', (username,))
    if cursor.fetchone():
        conn.close()
        return False, "Username is already registered."

    # Check duplicate email
    cursor.execute('SELECT id FROM users WHERE LOWER(email) = LOWER(?)', (email,))
    if cursor.fetchone():
        conn.close()
        return False, "Email address is already in use."

    password_hash = generate_password_hash(password)

    cursor.execute('''
        INSERT INTO users (username, email, password_hash, full_name, role, organization, station)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (username, email, password_hash, full_name, role, organization, station))

    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return True, user_id


def verify_user_credentials(identifier, password):
    user = get_user_by_email_or_username(identifier)
    if not user:
        return None, "No account found with this username or email."

    if not check_password_hash(user['password_hash'], password):
        return None, "Invalid password. Please check your credentials."

    # Update last login
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
    
    new_hash = generate_password_hash(new_password)
    conn = get_db_connection()
    conn.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_hash, user['id']))
    conn.commit()
    conn.close()
    return True, "Your password has been successfully reset. You may now log in."

