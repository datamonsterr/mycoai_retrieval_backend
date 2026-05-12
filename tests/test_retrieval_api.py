from fastapi.testclient import TestClient

from mycoai_retrieval_backend.app import app
from mycoai_retrieval_backend.retrieval import (
    BatchRetrievalRequest,
    BatchRetrievalResponse,
    BatchStrainResult,
    RankedSpecies,
    RetrievalResponse,
    SegmentQueryResult,
)
from mycoai_retrieval_backend.routes.retrieval import get_retrieval_pipeline


class StubPipeline:
    def query_image(self, request):
        image_ids = getattr(request, "image_ids", None) or [request.image_id]
        return RetrievalResponse(
            request_id="req-1",
            image_ids=image_ids,
            predictions=[
                RankedSpecies(species="Species A", score=1.0, neighbor_count=1)
            ],
            segments=[
                SegmentQueryResult(
                    segment_id="seg-1",
                    image_id=image_ids[0],
                    neighbors=[],
                )
            ],
            aggregation=request.aggregation,
            feature_extractor=getattr(request, "feature_extractor", None)
            or "EfficientNetV2B0",
            metrics={"response_time_ms": 1.0},
        )

    def query_strain(self, request):
        return self.query_image(request)

    def query_batch(self, request: BatchRetrievalRequest):
        return BatchRetrievalResponse(
            results=[
                BatchStrainResult(
                    strain_id=request.strains[0].strain_id,
                    result=self.query_strain(request.strains[0]),
                )
            ]
        )


client = TestClient(app)
app.dependency_overrides[get_retrieval_pipeline] = lambda: StubPipeline()


def test_image_endpoint_returns_predictions() -> None:
    response = client.post(
        "/api/retrieval/image",
        json={"image_id": "img-1", "k": 3, "env_strategy": "E2"},
    )

    assert response.status_code == 200
    assert response.json()["predictions"][0]["species"] == "Species A"


def test_batch_endpoint_returns_results() -> None:
    response = client.post(
        "/api/retrieval/batch",
        json={
            "strains": [
                {
                    "strain_id": "strain-1",
                    "image_ids": ["img-1"],
                    "k": 3,
                    "env_strategy": "E2",
                }
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["strain_id"] == "strain-1"
