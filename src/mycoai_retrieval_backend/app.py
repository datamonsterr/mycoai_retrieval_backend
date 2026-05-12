from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import get_settings

MEDIA_OPTIONS = [
    "MEA",
    "CYA",
    "YES",
    "DG18",
    "OA",
    "CREA",
    "PDA",
    "CMA",
    "SAB",
    "M40Y",
]
MAX_COLONIES_DEFAULT = None
MAX_COLONIES_MIN = 1
MAX_COLONIES_MAX = 10
MIN_IMAGE_DIMENSION = 256


class SingleImageUploadRequest(BaseModel):
    strain_id: str = Field(min_length=1, max_length=120)
    media: str = Field(pattern="^(?:" + "|".join(MEDIA_OPTIONS) + ")$")
    max_colonies: int | None = Field(
        default=None, ge=MAX_COLONIES_MIN, le=MAX_COLONIES_MAX
    )
    image_format: Literal["jpeg", "png", "tiff"]
    image_width: int = Field(ge=MIN_IMAGE_DIMENSION)
    image_height: int = Field(ge=MIN_IMAGE_DIMENSION)


class BatchColumnMapping(BaseModel):
    strain: str
    media: str
    max_colonies: str | None = None


class BatchDefaults(BaseModel):
    media: str = "MEA"
    max_colonies: int | None = None


class BatchTemplate(BaseModel):
    batch_name: str
    column_mapping: BatchColumnMapping
    defaults: BatchDefaults = BatchDefaults()
    output_format: Literal["csv"] = "csv"


class ReformatRequest(BaseModel):
    source_structure: str
    template: BatchTemplate


class ReformatResponse(BaseModel):
    batch_name: str
    template_path: str
    mapped_columns: dict[str, str]
    suggested_actions: list[str]


class BatchImageReviewItem(BaseModel):
    strain: str
    image_count: int
    removable_images: list[str]


class BatchReviewResponse(BaseModel):
    batch_name: str
    items: list[BatchImageReviewItem]
    progress: int


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
            "image_input": f"{settings.api_prefix}/image-input",
        }

    @app.get(f"{settings.api_prefix}/image-input/media-options", tags=["image-input"])
    def media_options() -> dict[str, list[str]]:
        return {"media": MEDIA_OPTIONS}

    @app.get(f"{settings.api_prefix}/image-input/template", tags=["image-input"])
    def template() -> BatchTemplate:
        return BatchTemplate(
            batch_name="batch_upload",
            column_mapping=BatchColumnMapping(
                strain="strain", media="media", max_colonies="max_colonies"
            ),
        )

    @app.post(f"{settings.api_prefix}/image-input/single-image", tags=["image-input"])
    def single_image(request: SingleImageUploadRequest) -> dict[str, object]:
        colony_limit = (
            request.max_colonies
            if request.max_colonies is not None
            else "model-threshold"
        )
        return {
            "strain_id": request.strain_id,
            "media": request.media,
            "max_colonies": colony_limit,
            "image_format": request.image_format,
            "preview_ready": True,
        }

    @app.post(f"{settings.api_prefix}/image-input/batch/reformat", tags=["image-input"])
    def reformat_batch(request: ReformatRequest) -> ReformatResponse:
        return ReformatResponse(
            batch_name=request.template.batch_name,
            template_path=f"{request.source_structure.rstrip('/')}/template.json",
            mapped_columns={
                "strain": request.template.column_mapping.strain,
                "media": request.template.column_mapping.media,
                "max_colonies": request.template.column_mapping.max_colonies or "",
            },
            suggested_actions=[
                "detect strain/media columns",
                "map optional max_colonies column",
                "emit batch_upload template",
            ],
        )

    @app.get(f"{settings.api_prefix}/image-input/batch/review", tags=["image-input"])
    def batch_review() -> BatchReviewResponse:
        return BatchReviewResponse(
            batch_name="batch_upload",
            progress=0,
            items=[
                BatchImageReviewItem(
                    strain="strain_001",
                    image_count=2,
                    removable_images=["image_02.jpg"],
                ),
                BatchImageReviewItem(
                    strain="strain_002",
                    image_count=1,
                    removable_images=["image_01.jpg"],
                ),
            ],
        )

    return app


app = create_app()
