from fastapi.testclient import TestClient

from mycoai_retrieval_backend.app import app

client = TestClient(app)


def test_mock_retrieval_returns_ranked_results() -> None:
    response = client.get("/api/retrieval/mock")

    assert response.status_code == 200
    body = response.json()
    assert body["strain"] == "Query-17"
    assert len(body["rankings"]) == 3
    assert body["rankings"][0]["rank"] == 1
    assert body["rankings"][0]["species"] == "Penicillium commune"
    assert body["rankings"][0]["score"] == 0.91
    assert body["query_details"]["k"] == 5
    assert body["query_details"]["aggregation"] == "weighted"
    assert body["query_details"]["environment_strategy"] == "E1"


def test_mock_retrieval_complies_with_openapi_schema() -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    paths = schema["paths"]
    assert "/api/retrieval/mock" in paths
