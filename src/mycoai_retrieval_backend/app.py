from fastapi import FastAPI, HTTPException, status

from .config import get_settings
from .feedback import (
    FeedbackBatchReview,
    FeedbackCreate,
    FeedbackReview,
    NotificationType,
)
from .feedback_store import get_store


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    @app.get("/health", tags=["health"])
    def healthcheck() -> dict[str, str]:
        return {
            "status": "ok",
            "service": settings.app_name,
            "environment": settings.environment,
        }

    @app.get("/", tags=["meta"])
    def root() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "docs": "/docs",
            "health": "/health",
        }

    @app.post(
        "/api/v1/feedback", status_code=status.HTTP_201_CREATED, tags=["feedback"]
    )
    def submit_feedback(payload: FeedbackCreate) -> dict[str, object]:
        record = get_store().submit(payload)
        return {"feedback": record.model_dump(mode="json")}

    @app.get("/api/v1/feedback", tags=["feedback"])
    def list_feedback(submitter_id: str) -> dict[str, object]:
        return {
            "feedback": [
                f.model_dump(mode="json")
                for f in get_store().list_by_submitter(submitter_id)
            ]
        }

    @app.get("/api/v1/feedback/inbox", tags=["feedback"])
    def inbox() -> dict[str, object]:
        return {
            "feedback": [f.model_dump(mode="json") for f in get_store().inbox()]
        }

    @app.patch("/api/v1/feedback/{feedback_id}", tags=["feedback"])
    def review_feedback(
        feedback_id: str, payload: FeedbackReview
    ) -> dict[str, object]:
        try:
            record = get_store().review_one(feedback_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="feedback not found") from exc

        msg = f"Your feedback was {payload.status.value}"
        get_store().notify(
            str(record.submitter_id),
            getattr(NotificationType, f"feedback_{payload.status.value}"),
            feedback_id,
            msg,
        )
        return {"feedback": record.model_dump(mode="json")}

    @app.post("/api/v1/feedback/batch", tags=["feedback"])
    def review_feedback_batch(payload: FeedbackBatchReview) -> dict[str, object]:
        records = get_store().review_batch(payload)
        return {
            "feedback": [record.model_dump(mode="json") for record in records]
        }

    @app.get("/api/v1/feedback/stats", tags=["feedback"])
    def feedback_stats() -> dict[str, object]:
        return {"stats": get_store().stats().model_dump(mode="json")}

    return app


app = create_app()
