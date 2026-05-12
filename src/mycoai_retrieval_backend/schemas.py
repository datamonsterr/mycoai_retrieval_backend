from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AggregationStrategy = Literal["weighted", "uni"]
EnvironmentStrategy = Literal["E1", "E2", "E3", "E4"]


class QueryNeighbor(BaseModel):
    image_id: str
    thumbnail_url: str
    species: str
    strain: str
    similarity: float = Field(ge=0, le=1)
    growth_medium: str


class QueryMediaNeighbors(BaseModel):
    media: str
    query_image_id: str
    neighbors: list[QueryNeighbor]


class RankedSpeciesResult(BaseModel):
    rank: int = Field(ge=1)
    species: str
    score: float = Field(ge=0, le=1)
    media_details: list[QueryMediaNeighbors]


class RetrievalQueryDetails(BaseModel):
    k: int = Field(ge=1, le=20)
    aggregation: AggregationStrategy
    environment_strategy: EnvironmentStrategy
    total_neighbors_queried: int = Field(ge=0)


class RetrievalQueryResponse(BaseModel):
    strain: str
    rankings: list[RankedSpeciesResult]
    query_details: RetrievalQueryDetails
