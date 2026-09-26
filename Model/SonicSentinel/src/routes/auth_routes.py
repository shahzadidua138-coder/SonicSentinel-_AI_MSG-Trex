# ============================================================
# SonicSentinel AI - Auth Routes (RBAC & Session Management)
# ============================================================
"""
Authentication & Authorization API Routes:
  - Login / Logout
  - Current User Profile & Role Info
  - User Registration (Admin Only)
"""

from flask import Blueprint, request, jsonify
from flask_login import login_user, logout_user, login_required, current_user

from src.extensions import db
from src.models.user import User, UserRole
from src.services.audit_service import AuditService

auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


@auth_bp.route("/login", methods=["POST"])
def login():
    """
    Authenticates user credentials and creates a Flask-Login session.
    """
    data = request.get_json() or {}
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    user = User.query.filter_by(username=username).first()

    if not user or not user.check_password(password):
        AuditService.log_action(
            action="LOGIN_FAILED",
            username=username,
            ip_address=request.remote_addr
        )
        return jsonify({"error": "Invalid credentials"}), 401

    if not user.is_active:
        return jsonify({"error": "Account is disabled"}), 403

    login_user(user, remember=data.get("remember", False))
    user.update_last_login()

    AuditService.log_action(
        action="LOGIN_SUCCESS",
        user_id=user.id,
        username=user.username,
        ip_address=request.remote_addr
    )

    return jsonify({
        "message": "Login successful",
        "user": user.to_dict()
    }), 200


@auth_bp.route("/logout", methods=["POST"])
@login_required
def logout():
    """
    Logs out current user session.
    """
    AuditService.log_action(
        action="LOGOUT",
        user_id=current_user.id,
        username=current_user.username,
        ip_address=request.remote_addr
    )
    logout_user()
    return jsonify({"message": "Logged out successfully"}), 200


@auth_bp.route("/me", methods=["GET"])
@login_required
def get_current_user_profile():
    """
    Returns current authenticated user profile & permissions.
    """
    return jsonify({"user": current_user.to_dict()}), 200


@auth_bp.route("/register", methods=["POST"])
def register_user():
    """
    User Registration Endpoint:
      - Admin accounts CANNOT self-register (Admin only logs in with pre-configured credentials).
      - Remaining 4 roles (Supervisor, Operator, Auditor, Monitor) can register and login.
    """
    data = request.get_json() or {}
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")
    role = data.get("role", UserRole.OPERATOR.value)

    if not username or not email or not password:
        return jsonify({"error": "Missing required fields (username, email, password)"}), 400

    # Admin accounts cannot self-register
    if role == UserRole.ADMIN.value:
        return jsonify({"error": "Admin accounts cannot be self-registered. Admin only logs in using pre-configured credentials."}), 403

    allowed_registration_roles = [
        UserRole.SUPERVISOR.value,
        UserRole.OPERATOR.value,
        UserRole.AUDITOR.value,
        UserRole.MONITOR.value
    ]
    if role not in allowed_registration_roles:
        return jsonify({"error": f"Invalid role '{role}'. Allowed registration roles: {allowed_registration_roles}"}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already taken"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 400

    user = User(
        username=username,
        email=email,
        full_name=data.get("full_name"),
        role=role
    )
    user.set_password(password)

    db.session.add(user)
    db.session.commit()

    AuditService.log_action(
        action="USER_REGISTERED",
        user_id=user.id,
        username=user.username,
        target_entity=f"User:{user.id}:{user.username}",
        ip_address=request.remote_addr
    )

    return jsonify({
        "message": "User registered successfully. You can now log in.",
        "user": user.to_dict()
    }), 201
