# ============================================================
# SonicSentinel AI - Routes Package
# ============================================================
from src.routes.auth_routes import auth_bp
from src.routes.audio_routes import audio_bp
from src.routes.prediction_routes import prediction_bp
from src.routes.alert_routes import alert_bp
from src.routes.review_routes import review_bp
from src.routes.dashboard_routes import dashboard_bp
from src.routes.model_routes import model_bp

__all__ = [
    "auth_bp",
    "audio_bp",
    "prediction_bp",
    "alert_bp",
    "review_bp",
    "dashboard_bp",
    "model_bp"
]
