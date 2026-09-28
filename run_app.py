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
    print("  Database: Neon PostgreSQL")
    print("==================================================================")

    # Initialize Neon PostgreSQL schema. Demo users/events are opt-in.
    database.init_db()

    port = int(os.environ.get('PORT', 5000))
    print(f"\n  System Status: ONLINE & ACTIVE")
    print(f"  Server Address: http://127.0.0.1:{port} (and http://localhost:{port})")
    print("  Create an account or login to get started.")
    print("==================================================================\n")

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
