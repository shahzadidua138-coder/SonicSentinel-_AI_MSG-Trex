# ============================================================
# SonicSentinel AI - Role-Based Access Control (RBAC) Utility
# ============================================================
"""
Provides RBAC decorators for enforcing user role permissions across API routes:
  - Admin
  - Supervisor
  - Operator
  - Auditor
  - Monitor
"""

from functools import wraps
from flask import jsonify
from flask_login import current_user
from src.models.user import UserRole


def role_required(*allowed_roles):
    """
    Decorator restricting access to specified roles.
    Example usage:
        @app.route("/api/admin-only")
        @login_required
        @role_required(UserRole.ADMIN.value, UserRole.SUPERVISOR.value)
        def admin_view():
            ...
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated:
                return jsonify({"error": "Authentication required"}), 401

            if current_user.role not in allowed_roles:
                return jsonify({
                    "error": "Forbidden: Insufficient role permissions",
                    "required_roles": list(allowed_roles),
                    "user_role": current_user.role
                }), 403

            return fn(*args, **kwargs)
        return wrapper
    return decorator


def admin_required(fn):
    """
    Convenience decorator for Admin-only routes.
    """
    return role_required(UserRole.ADMIN.value)(fn)


def supervisor_required(fn):
    """
    Convenience decorator for Admin or Supervisor routes.
    """
    return role_required(UserRole.ADMIN.value, UserRole.SUPERVISOR.value)(fn)


def operator_required(fn):
    """
    Convenience decorator for Admin, Supervisor, or Operator routes.
    """
    return role_required(
        UserRole.ADMIN.value,
        UserRole.SUPERVISOR.value,
        UserRole.OPERATOR.value
    )(fn)
