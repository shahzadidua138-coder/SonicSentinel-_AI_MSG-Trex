# ============================================================
# SonicSentinel AI - Review Queue Routes
# ============================================================
"""
API Endpoints for Manual Review Queue & Human-in-the-Loop Feedback:
  - Pending reviews queue inspection
  - Label overrides submission
  - Tagging audio events for dataset retraining
"""

from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user

from src.models.review import Review
from src.services.review_service import ReviewService
from src.services.audit_service import AuditService

review_bp = Blueprint("reviews", __name__, url_prefix="/api/reviews")
review_service = ReviewService()


@review_bp.route("/pending", methods=["GET"])

def list_pending_reviews():
    """
    Returns list of pending items requiring human operator verification.
    """
    limit = request.args.get("limit", 50, type=int)
    reviews = review_service.get_pending_reviews(limit=limit)
    return jsonify({
        "reviews": [r.to_dict() for r in reviews],
        "count": len(reviews)
    }), 200


@review_bp.route("/<int:review_id>/submit", methods=["POST"])

def submit_review_decision(review_id: int):
    """
    Submits human operator override or approval for a reviewed audio event.
    """
    data = request.get_json() or {}
    override_label = data.get("override_label")
    flag_for_retraining = data.get("flag_for_retraining", True)
    notes = data.get("notes")
    user_id = current_user.id if current_user.is_authenticated else 1

    try:
        review = review_service.submit_review(
            review_id=review_id,
            reviewer_id=user_id,
            override_label=override_label,
            flag_for_retraining=flag_for_retraining,
            notes=notes
        )

        AuditService.log_action(
            action="REVIEW_SUBMITTED",
            user_id=user_id,
            target_entity=f"Review:{review.id}",
            details={"override_label": override_label, "flagged_for_retraining": flag_for_retraining}
        )

        return jsonify({
            "message": "Review decision submitted successfully",
            "review": review.to_dict()
        }), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 404
