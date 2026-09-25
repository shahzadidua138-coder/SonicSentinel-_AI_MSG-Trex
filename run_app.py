"""
run_app.py - Single-Command Application Launcher
SonicSentinel AI - NextWave Acoustic Intelligence Platform
TechWiz 7 Competition Edition
"""

import os
import sys
import database
from app import app

if __name__ == '__main__':
    print("==================================================================")
    print("  SONICSENTINEL AI — NextWave Acoustic Intelligence Platform")
    print("  Theme: AcousticX Intelligence | TechWiz 7 Competition")
    print("  Budget: $0.00 (100% Free & Open-Source / Local-First)")
    print("==================================================================")

    # Initialize SQLite database and seed 11 mandatory demo events
    database.init_db()

    print("\n  System Status: ONLINE & ACTIVE")
    print("  Server Address: http://127.0.0.1:5000")
    print("\n  Default Operational Accounts:")
    print("    1. Administrator:         admin       | Password: Admin@123")
    print("    2. Security Operator:     operator    | Password: Operator@123")
    print("    3. Audio Reviewer:        reviewer    | Password: Reviewer@123")
    print("    4. Maintenance Operator:  maintenance | Password: Maint@123")
    print("    5. Normal User:           user        | Password: User@123")
    print("==================================================================\n")

    app.run(host='127.0.0.1', port=5000, debug=True)
