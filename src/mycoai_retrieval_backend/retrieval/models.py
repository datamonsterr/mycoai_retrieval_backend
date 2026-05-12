from typing import Literal

from pydantic import BaseModel, Field

AggregationStrategy = Literal["weighted", "uni", "manual_weighted"]
EnvironmentStrategy = str


class Neighbor(BaseModel):
    image_id: str
    segment_id: str | None = None
    parent_item_id: str | None = None
    species: str
    strain: str | None = None
    environment: str | None = None
    similarity: float = Field(ge=0.0)
    metadata: dict[str, object] = Field(default_factory=dict)


class RankedSpecies(BaseModel):
    species: str
    score: float
    neighbor_count: int


class SegmentQueryResult(BaseModel):
    segment_id: str
    image_id: str
    environment: str | None = None
    neighbors: list[Neighbor] = Field(default_factory=list)
    error: str | None = None


class RetrievalRequest(BaseModel):
    image_id: str
    k: int = Field(default=5, ge=1, le=100)
    env_strategy: EnvironmentStrategy = "E1"
    query_media: str | None = None
    aggregation: AggregationStrategy = "weighted"
    feature_extractor: str | None = None


class StrainRetrievalRequest(BaseModel):
    strain_id: str
    image_ids: list[str] = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=100)
    env_strategy: EnvironmentStrategy = "E1"
    aggregation: AggregationStrategy = "weighted"
    feature_extractor: str | None = None


class RetrievalResponse(BaseModel):
    request_id: str
    image_ids: list[str]
    predictions: list[RankedSpecies]
    segments: list[SegmentQueryResult]
    aggregation: AggregationStrategy
    feature_extractor: str
    errors: list[str] = Field(default_factory=list)
    metrics: dict[str, float | int] = Field(default_factory=dict)


class BatchRetrievalRequest(BaseModel):
    strains: list[StrainRetrievalRequest] = Field(min_length=1)


class BatchStrainResult(BaseModel):
    strain_id: str
    result: RetrievalResponse | None = None
    error: str | None = None


class BatchRetrievalResponse(BaseModel):
    results: list[BatchStrainResult]
    metrics: dict[str, float | int] = Field(default_factory=dict)
