# Default Pre-configured Admin Credentials:
#   Username: admin
#   Password: Admin@123456
#   Role: Admin (Admin only logs in; remaining 4 roles can register & login)

import os
import sys

# Ensure project root is in python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from flask import Flask, jsonify
from src.config import Config, DATABASE_PATH, UPLOAD_FOLDER, MODEL_DIR
from src.extensions import db, login_manager, cors
from src.models.user import User, UserRole
from src.routes import (
    auth_bp,
    audio_bp,
    prediction_bp,
    alert_bp,
    review_bp,
    dashboard_bp,
    model_bp
)


def create_app(config_class=Config) -> Flask:
    """
    Creates and configures an instance of the SonicSentinel AI Flask application.
    """
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure necessary folders exist
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Initialize Extensions
    db.init_app(app)
    login_manager.init_app(app)
    cors.init_app(app, resources={r"/api/*": {"origins": "*"}})

    # Auto-seed Admin User on app context if not exists
    with app.app_context():
        db.create_all()
        admin = User.query.filter_by(username="admin").first()
        if not admin:
            admin = User(
                username="admin",
                email="admin@sonicsentinel.ai",
                full_name="System Administrator",
                role=UserRole.ADMIN.value
            )
            admin.set_password("Admin@123456")
            db.session.add(admin)
            db.session.commit()

    # User Loader for Flask-Login
    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    @login_manager.unauthorized_handler
    def unauthorized():
        return jsonify({"error": "Authentication required"}), 401

    # Register Blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(audio_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(alert_bp)
    app.register_blueprint(review_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(model_bp)

    # Root / Health Route
    @app.route("/")
    def index():
        return jsonify({
            "system": "SonicSentinel AI - Audio Threat Intelligence System",
            "version": "1.0.0",
            "status": "Operational",
            "supported_models": ["SVM", "Random Forest", "XGBoost", "Google Teachable Machine"]
        })

    @app.route("/api/health")
    def health_check():
        return jsonify({
            "status": "healthy",
            "database": os.path.exists(DATABASE_PATH)
        }), 200

    # Custom CLI Commands
    @app.cli.command("init-db")
    def init_db_command():
        """Creates all database tables."""
        with app.app_context():
            db.create_all()
            print("Database initialized successfully.")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
