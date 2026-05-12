from fastapi.testclient import TestClient

from mycoai_retrieval_backend.app import MEDIA_OPTIONS, app

client = TestClient(app)


def test_media_options_returns_predefined_list() -> None:
    response = client.get("/api/image-input/media-options")

    assert response.status_code == 200
    data = response.json()
    assert data["media"] == MEDIA_OPTIONS
    assert len(data["media"]) == 10


def test_template_returns_default_batch_template() -> None:
    response = client.get("/api/image-input/template")

    assert response.status_code == 200
    data = response.json()
    assert data["batch_name"] == "batch_upload"
    assert data["column_mapping"]["strain"] == "strain"
    assert data["column_mapping"]["media"] == "media"


def test_single_image_valid_request() -> None:
    response = client.post(
        "/api/image-input/single-image",
        json={
            "strain_id": "test-strain-01",
            "media": "MEA",
            "image_format": "jpeg",
            "image_width": 512,
            "image_height": 384,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["strain_id"] == "test-strain-01"
    assert data["preview_ready"] is True


def test_single_image_with_max_colonies() -> None:
    response = client.post(
        "/api/image-input/single-image",
        json={
            "strain_id": "test-strain-02",
            "media": "CYA",
            "max_colonies": 3,
            "image_format": "png",
            "image_width": 512,
            "image_height": 512,
        },
    )

    assert response.status_code == 200
    assert response.json()["max_colonies"] == 3


def test_single_image_rejects_invalid_media() -> None:
    response = client.post(
        "/api/image-input/single-image",
        json={
            "strain_id": "bad",
            "media": "INVALID",
            "image_format": "jpeg",
            "image_width": 512,
            "image_height": 512,
        },
    )

    assert response.status_code == 422


def test_single_image_rejects_small_image() -> None:
    response = client.post(
        "/api/image-input/single-image",
        json={
            "strain_id": "small",
            "media": "MEA",
            "image_format": "jpeg",
            "image_width": 100,
            "image_height": 100,
        },
    )

    assert response.status_code == 422


def test_single_image_rejects_max_colonies_out_of_range() -> None:
    response = client.post(
        "/api/image-input/single-image",
        json={
            "strain_id": "range-test",
            "media": "MEA",
            "max_colonies": 99,
            "image_format": "jpeg",
            "image_width": 512,
            "image_height": 512,
        },
    )

    assert response.status_code == 422


def test_reformat_batch_returns_mapped_columns() -> None:
    response = client.post(
        "/api/image-input/batch/reformat",
        json={
            "source_structure": "my_arbitrary_folder/",
            "template": {
                "batch_name": "my-batch",
                "column_mapping": {"strain": "species_col", "media": "growth_media"},
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["batch_name"] == "my-batch"
    assert data["mapped_columns"]["strain"] == "species_col"
    assert len(data["suggested_actions"]) >= 1


def test_batch_review_returns_preview_items() -> None:
    response = client.get("/api/image-input/batch/review")

    assert response.status_code == 200
    data = response.json()
    assert data["batch_name"] == "batch_upload"
    assert data["progress"] == 0
    assert len(data["items"]) == 2
    assert data["items"][0]["image_count"] == 2
