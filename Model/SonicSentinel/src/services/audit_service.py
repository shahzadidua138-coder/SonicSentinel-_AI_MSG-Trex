# ============================================================
# SonicSentinel AI - Audit Trail Service
# ============================================================
"""
Audit Service logging all user activity, security events, 
and system parameter changes per SRS requirement.
"""

from typing import Dict, Any, Optional

from src.extensions import db
from src.models.audit_log import AuditLog


class AuditService:
    """
    Logs trackable system & user actions to AuditLog table.
    """

    @staticmethod
    def log_action(
        action: str,
        user_id: Optional[int] = None,
        username: Optional[str] = None,
        target_entity: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ) -> AuditLog:
        """
        Records an action entry into the audit trail log.
        """
        log_entry = AuditLog(
            user_id=user_id,
            username=username,
            action=action,
            target_entity=target_entity,
            details=details,
            ip_address=ip_address
        )

        db.session.add(log_entry)
        db.session.commit()
        return log_entry
