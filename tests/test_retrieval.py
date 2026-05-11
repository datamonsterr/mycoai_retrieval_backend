"""Tests for the retrieval pipeline — schemas, logic, and API endpoint."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from mycoai_retrieval_backend.app import app
from mycoai_retrieval_backend.retrieval import (
    aggregate_predictions,
    filter_siblings,
    resolve_environment,
)
from mycoai_retrieval_backend.schemas import RetrieveRequest, RetrieveResponse

client = TestClient(app)

def _valid_payload(**overrides: Any) -> dict[str, object]:
    base: dict[str, object] = {
        "strain": "DTO-001-A1",
        "images": [
            {
                "image_id": "uuid-1",
                "media": "MEA",
                "segments": [
                    {"segment_index": 0, "crop_path": "/tmp/fake_crop_1.jpg"},
                    {"segment_index": 1, "crop_path": "/tmp/fake_crop_2.jpg"},
                ],
            }
        ],
        "k": 5,
        "aggregation": "weighted",
        "environment_strategy": "E1",
    }
    base.update(overrides)
    return base


class TestRetrieveRequestSchema:
    def test_valid_minimal_payload(self) -> None:
        req = RetrieveRequest.model_validate(_valid_payload())
        assert req.strain == "DTO-001-A1"
        assert req.k == 5

    def test_k_min_1(self) -> None:
        req = RetrieveRequest.model_validate(_valid_payload(k=1))
        assert req.k == 1

    def test_k_max_20(self) -> None:
        req = RetrieveRequest.model_validate(_valid_payload(k=20))
        assert req.k == 20

    def test_k_below_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrieveRequest.model_validate(_valid_payload(k=0))

    def test_k_above_20_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrieveRequest.model_validate(_valid_payload(k=21))

    def test_aggregation_invalid_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrieveRequest.model_validate(_valid_payload(aggregation="invalid"))

    def test_aggregation_uni_accepted(self) -> None:
        req = RetrieveRequest.model_validate(_valid_payload(aggregation="uni"))
        assert req.aggregation == "uni"

    def test_e3_requires_medium(self) -> None:
        with pytest.raises(ValueError, match="e3_medium"):
            RetrieveRequest.model_validate(
                _valid_payload(environment_strategy="E3")
            )

    def test_e3_with_medium_accepted(self) -> None:
        req = RetrieveRequest.model_validate(
            _valid_payload(environment_strategy="E3", e3_medium="CYA")
        )
        assert req.e3_medium == "CYA"

    def test_e4_requires_exclude_medium(self) -> None:
        with pytest.raises(ValueError, match="e4_exclude_medium"):
            RetrieveRequest.model_validate(
                _valid_payload(environment_strategy="E4")
            )

    def test_e4_with_exclude_accepted(self) -> None:
        req = RetrieveRequest.model_validate(
            _valid_payload(environment_strategy="E4", e4_exclude_medium="CYA")
        )
        assert req.e4_exclude_medium == "CYA"

    def test_e1_multi_media_rejected(self) -> None:
        p = _valid_payload(
            environment_strategy="E1",
            images=[
                {"image_id": "u1", "media": "MEA", "segments": []},
                {"image_id": "u2", "media": "CYA", "segments": []},
            ],
        )
        with pytest.raises(ValueError, match="same medium"):
            RetrieveRequest.model_validate(p)

    def test_e2_multi_media_accepted(self) -> None:
        p = _valid_payload(
            environment_strategy="E2",
            images=[
                {"image_id": "u1", "media": "MEA", "segments": []},
                {"image_id": "u2", "media": "CYA", "segments": []},
            ],
        )
        req = RetrieveRequest.model_validate(p)
        assert req.environment_strategy == "E2"


class TestRetrieveResponseSchema:
    def test_serialisation(self) -> None:
        resp = RetrieveResponse(
            strain="DTO-001-A1",
            rankings=[
                {"rank": 1, "species": "Penicillium commune", "score": 0.87},
                {"rank": 2, "species": "Penicillium expansum", "score": 0.43},
            ],
            query_details={
                "k": 5,
                "aggregation": "weighted",
                "environment_strategy": "E1",
                "total_neighbors_queried": 15,
            },
        )
        data = json.loads(resp.model_dump_json())
        assert data["rankings"][0]["species"] == "Penicillium commune"


class TestFilterSiblings:
    def test_filters_same_parent(self) -> None:
        neighbours = [
            {"image_id": "A", "parent_id": "P1", "specy": "X"},
            {"image_id": "B", "parent_id": "P2", "specy": "Y"},
            {"image_id": "C", "parent_id": "P1", "specy": "Z"},
        ]
        result = filter_siblings(neighbours, "P1")
        assert len(result) == 1
        assert result[0]["image_id"] == "B"

    def test_no_parent_filter_passes_all(self) -> None:
        neighbours = [
            {"image_id": "A", "specy": "X"},
            {"image_id": "B", "specy": "Y"},
        ]
        result = filter_siblings(neighbours, "P1")
        assert len(result) == 2


class TestAggregatePredictions:
    def test_weighted_aggregation(self) -> None:
        results = [
            {
                "neighbours": [
                    {"specy": "A", "score": 0.9},
                    {"specy": "B", "score": 0.3},
                ]
            },
            {
                "neighbours": [
                    {"specy": "A", "score": 0.8},
                    {"specy": "C", "score": 0.2},
                ]
            },
        ]
        agg = aggregate_predictions(results, strategy="weighted")
        assert len(agg) == 3
        assert agg[0][0] == "A"
        assert round(agg[0][1], 3) == 0.425

    def test_uni_aggregation(self) -> None:
        results = [
            {
                "neighbours": [
                    {"specy": "A", "score": 0.9},
                    {"specy": "B", "score": 0.3},
                ]
            },
            {
                "neighbours": [
                    {"specy": "A", "score": 0.8},
                    {"specy": "C", "score": 0.2},
                ]
            },
        ]
        agg = aggregate_predictions(results, strategy="uni")
        assert agg[0][0] == "A"
        assert round(agg[0][1], 2) == 0.5
        assert round(agg[1][1], 2) == 0.25

    def test_empty_results(self) -> None:
        agg = aggregate_predictions([], strategy="weighted")
        assert agg == []

    def test_unknown_species_skipped(self) -> None:
        results = [
            {
                "neighbours": [
                    {"specy": "unknown", "score": 0.5},
                    {"specy": "A", "score": 0.9},
                ]
            }
        ]
        agg = aggregate_predictions(results, strategy="weighted")
        assert len(agg) == 1
        assert agg[0][0] == "A"


class TestResolveEnvironment:
    def test_e1_same_medium(self) -> None:
        inc, exc = resolve_environment("E1", "MEA")
        assert inc == "MEA"
        assert exc is None

    def test_e2_all_media(self) -> None:
        inc, exc = resolve_environment("E2", "MEA")
        assert inc is None
        assert exc is None

    def test_e3_specific_medium(self) -> None:
        inc, exc = resolve_environment("E3", "MEA", e3_medium="CYA")
        assert inc == "CYA"
        assert exc is None

    def test_e4_exclude_medium(self) -> None:
        inc, exc = resolve_environment("E4", "MEA", e4_exclude_medium="CYA")
        assert inc is None
        assert exc == "CYA"


class TestHealthEndpoint:
    def test_health_returns_ok(self) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_root_exposes_endpoints(self) -> None:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["health"] == "/health"


class TestRetrieveEndpoint:
    def test_retrieve_invalid_k(self) -> None:
        p = _valid_payload(k=0)
        response = client.post("/api/retrieve", json=p)
        assert response.status_code == 422

    def test_retrieve_invalid_aggregation(self) -> None:
        p = _valid_payload(aggregation="bad")
        response = client.post("/api/retrieve", json=p)
        assert response.status_code == 422

    def test_retrieve_e3_missing_medium(self) -> None:
        p = _valid_payload(environment_strategy="E3")
        response = client.post("/api/retrieve", json=p)
        assert response.status_code == 422


class TestFeatureExtraction:
    def test_hog_output_shape(self) -> None:
        from mycoai_retrieval_backend.retrieval import extract_hog_features

        dummy = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        feats = extract_hog_features(dummy)
        assert feats.ndim == 1
        assert feats.shape[0] > 0

        assert abs(float(np.linalg.norm(feats)) - 1.0) < 1e-6

    def test_l2_normalize(self) -> None:
        from mycoai_retrieval_backend.retrieval import _l2_normalize

        v = np.array([3.0, 4.0], dtype=np.float64)
        n = _l2_normalize(v)
        assert np.allclose(n, [0.6, 0.8])

    def test_l2_normalize_zero_vector(self) -> None:
        from mycoai_retrieval_backend.retrieval import _l2_normalize

        v = np.array([0.0, 0.0], dtype=np.float64)
        n = _l2_normalize(v)
        assert np.allclose(n, v)
