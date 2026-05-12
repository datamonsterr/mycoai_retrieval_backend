from __future__ import annotations

from fastapi import APIRouter

from ..schemas import (
    QueryMediaNeighbors,
    QueryNeighbor,
    RankedSpeciesResult,
    RetrievalQueryDetails,
    RetrievalQueryResponse,
)

router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


@router.get("/mock", response_model=RetrievalQueryResponse)
def mock_retrieval() -> RetrievalQueryResponse:
    neighbors = [
        QueryNeighbor(
            image_id="image-01",
            thumbnail_url="https://placehold.co/96x96/png",
            species="Penicillium commune",
            strain="PC-104",
            similarity=0.94,
            growth_medium="MEA",
        ),
        QueryNeighbor(
            image_id="image-02",
            thumbnail_url="https://placehold.co/96x96/png?text=PC",
            species="Penicillium expansum",
            strain="PE-221",
            similarity=0.87,
            growth_medium="CYA",
        ),
        QueryNeighbor(
            image_id="image-03",
            thumbnail_url="https://placehold.co/96x96/png?text=AS",
            species="Aspergillus niger",
            strain="AN-88",
            similarity=0.82,
            growth_medium="YES",
        ),
    ]

    return RetrievalQueryResponse(
        strain="Query-17",
        rankings=[
            RankedSpeciesResult(
                rank=1,
                species="Penicillium commune",
                score=0.91,
                media_details=[
                    QueryMediaNeighbors(
                        media="MEA",
                        query_image_id="query-mea-01",
                        neighbors=neighbors,
                    )
                ],
            ),
            RankedSpeciesResult(
                rank=2,
                species="Penicillium expansum",
                score=0.73,
                media_details=[
                    QueryMediaNeighbors(
                        media="CYA",
                        query_image_id="query-cya-01",
                        neighbors=neighbors[1:],
                    )
                ],
            ),
            RankedSpeciesResult(
                rank=3,
                species="Aspergillus niger",
                score=0.59,
                media_details=[
                    QueryMediaNeighbors(
                        media="YES",
                        query_image_id="query-yes-01",
                        neighbors=neighbors[2:],
                    )
                ],
            ),
        ],
        query_details=RetrievalQueryDetails(
            k=5,
            aggregation="weighted",
            environment_strategy="E1",
            total_neighbors_queried=15,
        ),
    )
