from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends
from qdrant_client import QdrantClient

from mycoai_retrieval_backend.config import Settings, get_settings
from mycoai_retrieval_backend.retrieval import (
    BatchRetrievalRequest,
    BatchRetrievalResponse,
    RetrievalPipeline,
    RetrievalRequest,
    RetrievalResponse,
    StrainRetrievalRequest,
    load_species_weights,
)

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


def get_qdrant_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> QdrantClient:
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
    )


def get_retrieval_pipeline(
    settings: Annotated[Settings, Depends(get_settings)],
    client: Annotated[QdrantClient, Depends(get_qdrant_client)],
) -> RetrievalPipeline:
    weights_path = settings.species_weights_path
    if not weights_path and settings.monorepo_root:
        weights_path = str(Path(settings.monorepo_root) / "species_weights.json")
    return RetrievalPipeline(
        client=client,
        collection_name=settings.qdrant_collection,
        default_feature_extractor=settings.feature_extractor,
        species_weights=load_species_weights(weights_path),
    )


@router.post("/image", response_model=RetrievalResponse)
def query_image(
    request: RetrievalRequest,
    pipeline: Annotated[RetrievalPipeline, Depends(get_retrieval_pipeline)],
) -> RetrievalResponse:
    return pipeline.query_image(request)


@router.post("/strain", response_model=RetrievalResponse)
def query_strain(
    request: StrainRetrievalRequest,
    pipeline: Annotated[RetrievalPipeline, Depends(get_retrieval_pipeline)],
) -> RetrievalResponse:
    return pipeline.query_strain(request)


@router.post("/batch", response_model=BatchRetrievalResponse)
def query_batch(
    request: BatchRetrievalRequest,
    pipeline: Annotated[RetrievalPipeline, Depends(get_retrieval_pipeline)],
) -> BatchRetrievalResponse:
    return pipeline.query_batch(request)
