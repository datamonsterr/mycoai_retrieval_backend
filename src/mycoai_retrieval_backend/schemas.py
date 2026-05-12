from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SegmentInput(BaseModel):
    segment_index: int
    crop_path: str


class ImageInput(BaseModel):
    image_id: str
    media: str
    segments: list[SegmentInput]


class RetrieveRequest(BaseModel):
    strain: str
    images: list[ImageInput]
    k: int = Field(default=5, ge=1, le=20)
    aggregation: Literal["weighted", "uni"] = "weighted"
    environment_strategy: Literal["E1", "E2", "E3", "E4"] = "E1"
    e3_medium: str | None = None
    e4_exclude_medium: str | None = None

    @model_validator(mode="after")
    def validate_strategy_params(self) -> RetrieveRequest:
        if self.environment_strategy == "E3" and not self.e3_medium:
            raise ValueError("e3_medium required when environment_strategy=E3")
        if self.environment_strategy == "E4" and not self.e4_exclude_medium:
            raise ValueError("e4_exclude_medium required when environment_strategy=E4")
        if self.environment_strategy == "E1":
            media = {img.media for img in self.images}
            if len(media) > 1:
                raise ValueError(
                    "E1 (same medium) requires all images to share one medium"
                )
        return self


class SpeciesRanking(BaseModel):
    rank: int
    species: str
    score: float


class QueryDetails(BaseModel):
    k: int
    aggregation: Literal["weighted", "uni"]
    environment_strategy: str
    total_neighbors_queried: int


class RetrieveResponse(BaseModel):
    strain: str
    rankings: list[SpeciesRanking]
    query_details: QueryDetails
