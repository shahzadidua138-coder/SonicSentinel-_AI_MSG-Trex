# ============================================================
# SonicSentinel AI - User Model
# ============================================================
import enum
import uuid
from datetime import datetime, timezone
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from src.extensions import db


class UserRole(str, enum.Enum):
    """Roles for role-based access control (RBAC)."""
    ADMIN = "Admin"
    SUPERVISOR = "Supervisor"
    OPERATOR = "Operator"
    AUDITOR = "Auditor"
    MONITOR = "Monitor"


class User(UserMixin, db.Model):
    """
    Represents a registered user in the SonicSentinel AI system.
    Supports RBAC with five distinct roles: Admin, Supervisor, Operator, Auditor, Monitor.
    """
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), unique=True, nullable=False, index=True, default=lambda: f"USR-{uuid.uuid4().hex[:8].upper()}")
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    full_name = db.Column(db.String(150), nullable=True)
    role = db.Column(db.String(20), default=UserRole.OPERATOR.value, nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    last_login = db.Column(db.DateTime, nullable=True)

    def set_password(self, password: str) -> None:
        """Hash and store a new password."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Verify a plaintext password against the stored hash."""
        return check_password_hash(self.password_hash, password)

    def update_last_login(self) -> None:
        """Updates last_login timestamp."""
        self.last_login = datetime.now(timezone.utc)
        db.session.commit()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None
        }

    def __repr__(self) -> str:
        return f"<User {self.username} [{self.role}]>"
