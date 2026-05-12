from fastapi.testclient import TestClient

from mycoai_retrieval_backend.app import app

client = TestClient(app)


class TestSpeciesCRUD:
    def test_create_species(self) -> None:
        response = client.post(
            "/api/data-management/species",
            json={"name": "Amanita muscaria", "description": "Fly agaric"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Amanita muscaria"
        assert data["description"] == "Fly agaric"
        assert data["species_id"]
        assert data["created_at"]
        assert data["is_archived"] is False
        assert data["strain_count"] == 0
        assert data["image_count"] == 0

    def test_create_species_duplicate_name_is_rejected(self) -> None:
        client.post(
            "/api/data-management/species",
            json={"name": "Boletus edulis"},
        )
        response = client.post(
            "/api/data-management/species",
            json={"name": "boletus edulis"},
        )
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    def test_list_species(self) -> None:
        client.post("/api/data-management/species", json={"name": "Species A"})
        client.post("/api/data-management/species", json={"name": "Species B"})
        response = client.get("/api/data-management/species")
        assert response.status_code == 200
        data = response.json()
        names = {s["name"] for s in data}
        assert "Species A" in names
        assert "Species B" in names

    def test_update_species_name(self) -> None:
        r = client.post("/api/data-management/species", json={"name": "Old Name"})
        species_id = r.json()["species_id"]
        response = client.patch(
            f"/api/data-management/species/{species_id}",
            json={"name": "New Name"},
        )
        assert response.status_code == 200
        assert response.json()["name"] == "New Name"

    def test_archive_species_cascades(self) -> None:
        r = client.post(
            "/api/data-management/species",
            json={"name": "Archivable"},
        )
        s = r.json()
        client.post(
            "/api/data-management/strains",
            json={"name": "Strain-1", "species_id": s["species_id"]},
        )
        archive_resp = client.post(
            f"/api/data-management/species/{s['species_id']}/archive",
        )
        assert archive_resp.status_code == 200
        species_data = archive_resp.json()
        assert species_data["is_archived"] is True
        strains_data = client.get(
            f"/api/data-management/strains?species_id={s['species_id']}&include_archived=true",
        ).json()
        for strain in strains_data:
            assert strain["is_archived"] is True

    def test_restore_species(self) -> None:
        r = client.post(
            "/api/data-management/species",
            json={"name": "Restorable"},
        )
        species_id = r.json()["species_id"]
        client.post(f"/api/data-management/species/{species_id}/archive")
        response = client.post(
            f"/api/data-management/species/{species_id}/restore",
        )
        assert response.status_code == 200
        assert response.json()["is_archived"] is False

    def test_delete_species_permanent(self) -> None:
        r = client.post(
            "/api/data-management/species",
            json={"name": "Deletable"},
        )
        species_id = r.json()["species_id"]
        response = client.delete(f"/api/data-management/species/{species_id}")
        assert response.status_code == 204
        response = client.get("/api/data-management/species")
        names = {s["name"] for s in response.json()}
        assert "Deletable" not in names

    def test_species_not_found_returns_404(self) -> None:
        response = client.patch(
            "/api/data-management/species/nonexistent",
            json={"name": "x"},
        )
        assert response.status_code == 404


class TestStrainCRUD:
    def test_create_strain(self) -> None:
        r = client.post(
            "/api/data-management/species",
            json={"name": "Host Species"},
        )
        species_id = r.json()["species_id"]
        response = client.post(
            "/api/data-management/strains",
            json={"name": "Strain Alpha", "species_id": species_id},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Strain Alpha"
        assert data["species_id"] == species_id
        assert data["source"] == "user_upload"
        assert data["image_count"] == 0

    def test_create_strain_for_missing_species_returns_404(self) -> None:
        response = client.post(
            "/api/data-management/strains",
            json={"name": "Orphan", "species_id": "no-such-species"},
        )
        assert response.status_code == 404

    def test_list_strains_filtered_by_species(self) -> None:
        r1 = client.post(
            "/api/data-management/species",
            json={"name": "S1"},
        )
        r2 = client.post(
            "/api/data-management/species",
            json={"name": "S2"},
        )
        client.post(
            "/api/data-management/strains",
            json={"name": "Strain-S1", "species_id": r1.json()["species_id"]},
        )
        client.post(
            "/api/data-management/strains",
            json={"name": "Strain-S2", "species_id": r2.json()["species_id"]},
        )
        response = client.get(
            f"/api/data-management/strains?species_id={r1.json()['species_id']}",
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "Strain-S1"


class TestImageCRUD:
    def test_create_image(self) -> None:
        r = client.post(
            "/api/data-management/species",
            json={"name": "Img Host"},
        )
        s = client.post(
            "/api/data-management/strains",
            json={"name": "Strain-I", "species_id": r.json()["species_id"]},
        )
        strain_id = s.json()["strain_id"]
        response = client.post(
            "/api/data-management/images",
            json={
                "strain_id": strain_id,
                "media": "PDA",
                "file_path": "/data/img001.jpg",
                "segments": [
                    {"segment_index": 0, "bbox": {"x": 1, "y": 2, "w": 100, "h": 200}}
                ],
                "indexed_in_qdrant": True,
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["media"] == "PDA"
        assert data["file_path"] == "/data/img001.jpg"
        assert len(data["segments"]) == 1
        assert data["indexed_in_qdrant"] is True

    def test_create_image_missing_strain_returns_404(self) -> None:
        response = client.post(
            "/api/data-management/images",
            json={"strain_id": "no-such-strain", "media": "PDA", "file_path": "/x.jpg"},
        )
        assert response.status_code == 404

    def test_batch_create_images_with_species_classification(self) -> None:
        r = client.post("/api/data-management/species", json={"name": "Batch Host"})
        response = client.post(
            "/api/data-management/images/batch",
            json=[
                {
                    "species_id": r.json()["species_id"],
                    "strain_name": "Batch-1",
                    "media": "PDA",
                    "file_path": "/batch/1.jpg",
                },
                {
                    "species_id": r.json()["species_id"],
                    "strain_name": "Batch-2",
                    "media": "MEA",
                    "file_path": "/batch/2.jpg",
                },
            ],
        )
        assert response.status_code == 201
        data = response.json()
        assert data["created_strains"] == 2
        assert data["created_images"] == 2
        assert all(image["indexed_in_qdrant"] for image in data["images"])

    def test_list_images_filtered_by_media(self) -> None:
        r = client.post("/api/data-management/species", json={"name": "Media Host"})
        s = client.post(
            "/api/data-management/strains",
            json={"name": "Strain-M", "species_id": r.json()["species_id"]},
        )
        sid = s.json()["strain_id"]
        client.post(
            "/api/data-management/images",
            json={"strain_id": sid, "media": "PDA", "file_path": "/a.jpg"},
        )
        client.post(
            "/api/data-management/images",
            json={"strain_id": sid, "media": "MEA", "file_path": "/b.jpg"},
        )
        response = client.get(
            f"/api/data-management/images?media=PDA&media=MEA&species_id={r.json()['species_id']}",
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2


class TestDashboard:
    def test_dashboard_counts(self) -> None:
        response = client.get("/api/data-management/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert "total_images" in data
        assert "total_strains" in data
        assert "total_species" in data
        assert "total_media_types" in data
        assert isinstance(data["images_per_species"], list)
        assert isinstance(data["images_per_medium"], list)
        assert isinstance(data["timeline"], list)
        assert "learned_count" in data
        assert "pending_count" in data
        assert "archived_since_last_training" in data


class TestAuditLog:
    def test_audit_log_captures_create_species(self) -> None:
        client.post(
            "/api/data-management/species",
            json={"name": "AuditedSpecies"},
        )
        response = client.get("/api/data-management/audit-log")
        assert response.status_code == 200
        logs = response.json()
        assert len(logs) >= 1
        assert logs[0]["action"] == "create"
        assert logs[0]["entity_type"] == "species"


class TestTrash:
    def test_empty_trash_permanently_deletes_archived_items(self) -> None:
        r = client.post("/api/data-management/species", json={"name": "ToBeTrashed"})
        species_id = r.json()["species_id"]
        client.post(f"/api/data-management/species/{species_id}/archive")
        response = client.delete("/api/data-management/trash")
        assert response.status_code == 204
        get_resp = client.get("/api/data-management/species?include_archived=true")
        names = {s["name"] for s in get_resp.json()}
        assert "ToBeTrashed" not in names
