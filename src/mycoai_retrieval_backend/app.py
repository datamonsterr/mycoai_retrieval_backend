from fastapi import FastAPI

from .config import get_settings
from .training_router import router as training_router


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.include_router(training_router)

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

    return app


app = create_app()
