from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID

from .feedback import (
    AuditLogEntry,
    FeedbackBatchReview,
    FeedbackCreate,
    FeedbackRecord,
    FeedbackReview,
    FeedbackSource,
    FeedbackStats,
    FeedbackStatus,
    Notification,
    NotificationType,
    ReindexTask,
)


class FeedbackStore:
    """In-memory feedback store for development and testing."""

    def __init__(self) -> None:
        self.feedback: list[FeedbackRecord] = []
        self.notifications: list[Notification] = []
        self.audit_log: list[AuditLogEntry] = []
        self.reindex_tasks: list[ReindexTask] = []
        self._audit_seq = 0

    def submit(self, payload: FeedbackCreate) -> FeedbackRecord:
        record = FeedbackRecord(
            submitter_id=payload.submitter_id,
            source=payload.source,
            suggested_species=payload.suggested_species,
            description=payload.description,
            query_strain=payload.query_strain,
            result_id=payload.result_id,
            image_id=payload.image_id,
            predicted_species=payload.predicted_species,
            supporting_image_url=payload.supporting_image_url,
        )
        self.feedback.append(record)
        self._log_action(
            user_id=str(payload.submitter_id),
            action="submit_feedback",
            entity_type="feedback",
            entity_id=str(record.id),
            changes={"status": "pending"},
        )
        return record

    def list_by_submitter(self, submitter_id: str) -> list[FeedbackRecord]:
        return [f for f in self.feedback if str(f.submitter_id) == submitter_id]

    def inbox(self) -> list[FeedbackRecord]:
        pending = [f for f in self.feedback if f.status == FeedbackStatus.pending]
        return sorted(pending, key=lambda f: f.submitted_at, reverse=True)

    def get(self, feedback_id: str) -> FeedbackRecord | None:
        for f in self.feedback:
            if str(f.id) == feedback_id:
                return f
        return None

    def review_one(self, feedback_id: str, review: FeedbackReview) -> FeedbackRecord:
        record = self._must_get(feedback_id)
        record.status = review.status
        record.reviewer_id = review.reviewer_id
        record.review_note = review.review_note
        record.reviewed_at = datetime.now(UTC)

        self._log_action(
            user_id=str(review.reviewer_id),
            action=f"{review.status.value}_feedback",
            entity_type="feedback",
            entity_id=str(record.id),
            changes={"status": review.status.value},
        )

        if review.status == FeedbackStatus.accepted:
            self._enqueue_reindex(record)

        return record

    def review_batch(self, batch: FeedbackBatchReview) -> list[FeedbackRecord]:
        results: list[FeedbackRecord] = []
        for fid in batch.feedback_ids:
            review = FeedbackReview(
                reviewer_id=batch.reviewer_id,
                status=batch.status,
                review_note=batch.review_note,
            )
            results.append(self.review_one(str(fid), review))
        return results

    def notify(
        self,
        user_id: str,
        type_: NotificationType,
        feedback_id: str,
        message: str,
    ) -> None:
        self.notifications.append(
            Notification(
                user_id=UUID(user_id),
                type=type_,
                feedback_id=UUID(feedback_id),
                message=message,
            )
        )

    def notifications_for(self, user_id: str) -> list[Notification]:
        return [
            n
            for n in self.notifications
            if str(n.user_id) == user_id and not n.read
        ]

    def mark_notification_read(self, notification_id: str) -> None:
        for n in self.notifications:
            if str(n.id) == notification_id:
                n.read = True
                return

    def stats(self) -> FeedbackStats:
        reviewed = [f for f in self.feedback if f.status != FeedbackStatus.pending]
        accepted = sum(1 for f in reviewed if f.status == FeedbackStatus.accepted)
        return FeedbackStats(
            total_by_source={
                source: sum(1 for f in self.feedback if f.source == source)
                for source in (
                    FeedbackSource.query_result,
                    FeedbackSource.database_review,
                )
            },
            acceptance_rate=accepted / len(reviewed) if reviewed else 0.0,
            average_review_seconds=None,
            most_misclassified_species=[],
            feedback_by_strain=[],
        )

    def _must_get(self, feedback_id: str) -> FeedbackRecord:
        record = self.get(feedback_id)
        if record is None:
            raise KeyError(feedback_id)
        return record

    def _enqueue_reindex(self, record: FeedbackRecord) -> None:
        task = ReindexTask(
            feedback_id=record.id,
            strain=record.query_strain or "unknown",
        )
        self.reindex_tasks.append(task)
        record.reindex_task_id = task.id

    def _log_action(
        self,
        user_id: str,
        action: str,
        entity_type: str,
        entity_id: str,
        changes: dict[str, object],
    ) -> None:
        self._audit_seq += 1
        self.audit_log.append(
            AuditLogEntry(
                id=self._audit_seq,
                user_id=UUID(str(user_id)),
                action=action,
                entity_type=entity_type,
                entity_id=UUID(str(entity_id)),
                changes=changes,
            )
        )


_store: FeedbackStore | None = None


def get_store() -> FeedbackStore:
    global _store
    if _store is None:
        _store = FeedbackStore()
    return _store


def reset_store() -> None:
    global _store
    _store = None


def inject_store(factory: Callable[[], FeedbackStore]) -> None:
    global _store
    _store = factory()
