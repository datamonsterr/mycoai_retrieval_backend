from typing import Any

import pytest

from mycoai_retrieval_backend.retrieval import RetrievalPipeline, RetrievalRequest


class FakePoint:
    def __init__(self, point_id: str, score: float, payload: dict[str, object]):
        self.id = point_id
        self.score = score
        self.payload = payload


class FakeResponse:
    def __init__(self, points: list[FakePoint]):
        self.points = points


class FakeClient:
    def __init__(self) -> None:
        self.query_filters: list[Any] = []

    def query_points(self, **kwargs: Any) -> FakeResponse:
        self.query_filters.append(kwargs["query_filter"])
        return FakeResponse(
            [
                FakePoint(
                    "sibling",
                    0.99,
                    {
                        "image_id": "sibling",
                        "parent_id": "parent-1",
                        "specy": "Species A",
                        "environment": "MEA",
                    },
                ),
                FakePoint(
                    "neighbor-1",
                    0.9,
                    {
                        "image_id": "neighbor-1",
                        "parent_id": "parent-2",
                        "specy": "Species B",
                        "environment": "MEA",
                    },
                ),
                FakePoint(
                    "neighbor-2",
                    0.8,
                    {
                        "image_id": "neighbor-2",
                        "parent_id": "parent-3",
                        "specy": "Species B",
                        "environment": "MEA",
                    },
                ),
            ]
        )


class FakeSegmentStore:
    def get_segments(self, image_id: str) -> list[dict[str, object]]:
        return [
            {
                "segment_id": f"{image_id}-seg-1",
                "image_id": image_id,
                "parent_item_id": "parent-1",
                "environment": "MEA",
                "vector": [0.1, 0.2, 0.3],
                "metadata": {},
            }
        ]


def test_query_image_filters_siblings_and_aggregates_predictions() -> None:
    client = FakeClient()
    pipeline = RetrievalPipeline(
        client=client,  # type: ignore[arg-type]
        collection_name="collection",
        default_feature_extractor="EfficientNetV2B0",
        segment_store=FakeSegmentStore(),
    )

    result = pipeline.query_image(
        RetrievalRequest(image_id="image-1", k=5, env_strategy="E1", query_media="MEA")
    )

    assert result.image_ids == ["image-1"]
    assert result.errors == []
    assert result.predictions[0].species == "Species B"
    assert result.predictions[0].score == pytest.approx(1.7)
    assert result.segments[0].neighbors[0].image_id == "neighbor-1"
    assert client.query_filters[0].must[0].match.value == "MEA"


def test_query_image_reports_missing_segments() -> None:
    class EmptySegmentStore:
        def get_segments(self, image_id: str) -> list[dict[str, object]]:
            return []

    pipeline = RetrievalPipeline(
        client=FakeClient(),  # type: ignore[arg-type]
        collection_name="collection",
        default_feature_extractor="EfficientNetV2B0",
        segment_store=EmptySegmentStore(),
    )

    result = pipeline.query_image(RetrievalRequest(image_id="image-1"))

    assert result.predictions == []
    assert result.errors == ["image-1: no segments found"]
    assert result.metrics["success"] == 0
