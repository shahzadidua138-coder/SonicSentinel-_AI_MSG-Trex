# ============================================================
# SonicSentinel AI - Database Initialization Script
# ============================================================
"""
Script to create SQLite database tables and default initial users.
"""

import os
import sys

# Ensure root directory is on python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.app import create_app
from src.extensions import db
from src.models.user import User, UserRole
from src.models.model_version import ModelVersion


def init_database():
    app = create_app()
    with app.app_context():
        print("Creating database schema...")
        db.create_all()

        # Check if default admin exists
        admin = User.query.filter_by(username="admin").first()
        if not admin:
            print("Creating default admin account...")
            admin = User(
                username="admin",
                email="admin@sonicsentinel.ai",
                full_name="System Administrator",
                role=UserRole.ADMIN.value
            )
            admin.set_password("Admin@123456")
            db.session.add(admin)

        # Check if default operator exists
        operator = User.query.filter_by(username="operator").first()
        if not operator:
            print("Creating default operator account...")
            operator = User(
                username="operator",
                email="operator@sonicsentinel.ai",
                full_name="Security Operator",
                role=UserRole.OPERATOR.value
            )
            operator.set_password("Operator@123456")
            db.session.add(operator)

        # Register default model version records
        py_model_v1 = ModelVersion.query.filter_by(version_name="v1.0.0", algorithm="XGBoost").first()
        if not py_model_v1:
            py_model_v1 = ModelVersion(
                version_name="v1.0.0",
                algorithm="XGBoost",
                framework="Python-Scikit/XGBoost",
                features_dimensionality=373,
                is_active=True,
                accuracy=0.9650,
                precision=0.9610,
                recall=0.9630,
                f1_score=0.9620,
                description="Trained Python XGBoost Classifier on 373-dim audio features."
            )
            db.session.add(py_model_v1)

        gtm_model_v1 = ModelVersion.query.filter_by(version_name="v1.0.0", algorithm="Google Teachable Machine").first()
        if not gtm_model_v1:
            gtm_model_v1 = ModelVersion(
                version_name="v1.0.0",
                algorithm="Google Teachable Machine",
                framework="Google Teachable Machine",
                features_dimensionality=128,
                is_active=True,
                accuracy=0.9420,
                precision=0.9380,
                recall=0.9400,
                f1_score=0.9390,
                description="Google Teachable Machine audio classifier model."
            )
            db.session.add(gtm_model_v1)

        db.session.commit()
        print("Database initialization complete!")


if __name__ == "__main__":
    init_database()
