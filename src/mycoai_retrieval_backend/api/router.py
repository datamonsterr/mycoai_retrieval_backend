from fastapi import APIRouter

from . import auth, dashboard, feedback, images, retrieval, species, training

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth.router)
api_router.include_router(dashboard.router)
api_router.include_router(feedback.router)
api_router.include_router(images.router)
api_router.include_router(retrieval.router)
api_router.include_router(species.router)
api_router.include_router(training.router)
