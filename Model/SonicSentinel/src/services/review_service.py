# ============================================================
# SonicSentinel AI - Human Review Queue Service
# ============================================================
"""
Review Service handling manual human reviews:
  - Fetching pending reviews (uncertain predictions, model mismatches, flagged alerts)
  - Submitting human label overrides
  - Tagging audio events for future model retraining dataset updates
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from src.extensions import db
from src.models.review import Review, ReviewStatus
from src.models.prediction import Prediction
from src.models.audio_event import AudioEvent


class ReviewService:
    """
    Manages operator manual reviews, overrides, and retraining dataset curation.
    """

    def create_review_task(
        self,
        audio_event_id: int,
        prediction_id: int,
        reason: str = "Uncertain prediction"
    ) -> Review:
        """
        Creates a new Review queue item for human inspection.
        """
        existing = Review.query.filter_by(
            audio_event_id=audio_event_id,
            prediction_id=prediction_id
        ).first()

        if existing:
            return existing

        review = Review(
            audio_event_id=audio_event_id,
            prediction_id=prediction_id,
            status=ReviewStatus.PENDING.value,
            notes=f"Flagged for review: {reason}"
        )

        db.session.add(review)
        db.session.commit()
        return review

    def submit_review(
        self,
        review_id: int,
        reviewer_id: int,
        override_label: Optional[str] = None,
        flag_for_retraining: bool = True,
        notes: Optional[str] = None
    ) -> Review:
        """
        Submits operator verification or label override for a review task.
        """
        review = db.session.get(Review, review_id)
        if not review:
            raise ValueError(f"Review with ID {review_id} not found.")

        review.reviewer_id = reviewer_id
        review.reviewed_at = datetime.now(timezone.utc)
        review.flagged_for_retraining = flag_for_retraining

        if notes:
            review.notes = notes

        if override_label:
            review.override_sound_class = override_label
            review.is_overridden = True
            review.status = ReviewStatus.OVERRIDDEN.value
        else:
            review.is_overridden = False
            review.status = ReviewStatus.APPROVED.value

        # Update associated AudioEvent status
        if review.audio_event:
            review.audio_event.status = "Reviewed"

        db.session.commit()
        return review

    def get_pending_reviews(self, limit: int = 50) -> List[Review]:
        """
        Retrieves pending review items sorted by creation time.
        """
        return Review.query.filter_by(
            status=ReviewStatus.PENDING.value
        ).order_by(Review.created_at.asc()).limit(limit).all()
