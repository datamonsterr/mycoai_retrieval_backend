from fastapi import APIRouter, HTTPException, status

from ..config import get_settings
from ..retrieval import get_qdrant_client, retrieve
from ..schemas import (
    QueryDetails,
    RetrieveRequest,
    RetrieveResponse,
    SpeciesRanking,
)

router = APIRouter(tags=["retrieval"])


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Retrieve species predictions for a strain",
    description=(
        "Submit segmented colony images and receive ranked species predictions."
    ),
)
def retrieve_endpoint(body: RetrieveRequest) -> RetrieveResponse:
    settings = get_settings()

    images: list[dict[str, object]] = []
    for img in body.images:
        images.append(
            {
                "image_id": img.image_id,
                "media": img.media,
                "segments": [
                    {"segment_index": s.segment_index, "crop_path": s.crop_path}
                    for s in img.segments
                ],
            }
        )

    try:
        client = get_qdrant_client(settings.qdrant_url, settings.qdrant_api_key)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Qdrant connection failed: {exc}",
        ) from exc

    try:
        result = retrieve(
            client=client,
            collection_name=settings.qdrant_collection,
            strain=body.strain,
            images=images,
            k=body.k,
            aggregation=body.aggregation,
            environment_strategy=body.environment_strategy,
            e3_medium=body.e3_medium,
            e4_exclude_medium=body.e4_exclude_medium,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    rankings = [
        SpeciesRanking(rank=r["rank"], species=r["species"], score=r["score"])
        for r in result["rankings"]
    ]
    details = QueryDetails(**result["query_details"])

    return RetrieveResponse(
        strain=result["strain"], rankings=rankings, query_details=details
    )
