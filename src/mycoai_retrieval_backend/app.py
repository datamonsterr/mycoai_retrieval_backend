from __future__ import annotations

import json
import sqlite3
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from .config import get_settings
from .store import DataStore, utc_now


class SpeciesCreate(BaseModel):
    name: str = Field(min_length=1)
    description: str | None = None
    taxonomic_info: str | None = None
    reference_images: list[str] = Field(default_factory=list)


class SpeciesUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    taxonomic_info: str | None = None
    reference_images: list[str] | None = None
    is_archived: bool | None = None


class SpeciesOut(BaseModel):
    species_id: str
    name: str
    description: str | None
    taxonomic_info: str | None
    reference_images: list[str]
    created_at: str
    updated_at: str
    is_archived: bool
    archived_alias_of: str | None = None
    strain_count: int
    image_count: int


class StrainCreate(BaseModel):
    name: str = Field(min_length=1)
    species_id: str
    source: str = Field(default="user_upload")


class StrainOut(BaseModel):
    strain_id: str
    name: str
    species_id: str
    created_at: str
    is_archived: bool
    source: str
    image_count: int


class ImageCreate(BaseModel):
    strain_id: str
    media: str
    file_path: str
    segments: list[dict[str, object]] = Field(default_factory=list)
    indexed_in_qdrant: bool = False


class ImageOut(BaseModel):
    image_id: str
    strain_id: str
    media: str
    file_path: str
    segments: list[dict[str, object]]
    indexed_in_qdrant: bool
    created_at: str
    is_archived: bool


class BatchImageCreate(BaseModel):
    species_id: str
    strain_name: str = Field(min_length=1)
    media: str
    file_path: str
    segments: list[dict[str, object]] = Field(default_factory=list)
    indexed_in_qdrant: bool = True
    source: str = Field(default="curated_primary")


class BatchUploadOut(BaseModel):
    created_strains: int
    created_images: int
    images: list[ImageOut]


class DashboardOut(BaseModel):
    total_images: int
    total_strains: int
    total_species: int
    total_media_types: int
    images_per_species: list[dict[str, int | str]]
    images_per_medium: list[dict[str, int | str]]
    learned_count: int
    pending_count: int
    timeline: list[dict[str, int | str]]
    archived_since_last_training: int


class AuditLogOut(BaseModel):
    audit_id: str
    action: str
    entity_type: str
    entity_id: str
    detail: str
    created_at: str


def get_store(request: Request) -> DataStore:
    return request.app.state.store


StoreDep = Depends(get_store)
MediaQuery = Annotated[list[str], Query(default_factory=list)]


def row_to_species(
    row: sqlite3.Row,
    strain_count: int,
    image_count: int,
) -> SpeciesOut:
    return SpeciesOut(
        species_id=row["species_id"],
        name=row["name"],
        description=row["description"],
        taxonomic_info=row["taxonomic_info"],
        reference_images=json.loads(row["reference_images"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        is_archived=bool(row["is_archived"]),
        archived_alias_of=row["archived_alias_of"],
        strain_count=strain_count,
        image_count=image_count,
    )


def slug_key(name: str) -> str:
    return name.casefold().strip()


def ensure_unique_species(
    store: DataStore, name: str, exclude_species_id: str | None = None
) -> None:
    params: list[object] = [slug_key(name)]
    sql = "SELECT species_id FROM species WHERE name_key = ?"
    if exclude_species_id is not None:
        sql += " AND species_id != ?"
        params.append(exclude_species_id)
    row = store.query_one(sql, tuple(params))
    if row is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Species name already exists"
        )


def get_species_or_404(store: DataStore, species_id: str) -> sqlite3.Row:
    row = store.query_one("SELECT * FROM species WHERE species_id = ?", (species_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="Species not found")
    return row


def get_strain_or_404(store: DataStore, strain_id: str) -> sqlite3.Row:
    row = store.query_one("SELECT * FROM strains WHERE strain_id = ?", (strain_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="Strain not found")
    return row


def species_counters(store: DataStore, species_id: str) -> tuple[int, int]:
    strain_count = store.query_one(
        "SELECT COUNT(*) AS c FROM strains WHERE species_id = ?",
        (species_id,),
    )["c"]
    image_count = store.query_one(
        """
        SELECT COUNT(*) AS c
        FROM images
        JOIN strains ON strains.strain_id = images.strain_id
        WHERE strains.species_id = ?
        """,
        (species_id,),
    )["c"]
    return int(strain_count), int(image_count)


def make_router() -> APIRouter:
    router = APIRouter(prefix="/api/data-management", tags=["data-management"])

    @router.get("/species", response_model=list[SpeciesOut])
    def list_species(
        store: Annotated[DataStore, StoreDep],
        include_archived: bool = False,
    ) -> list[SpeciesOut]:
        rows = store.query(
            "SELECT * FROM species WHERE is_archived = ? OR ?",
            (0 if not include_archived else 1, 1 if include_archived else 0),
        )
        return [
            row_to_species(row, *species_counters(store, row["species_id"]))
            for row in rows
        ]

    @router.post("/species", response_model=SpeciesOut, status_code=201)
    def create_species(
        payload: SpeciesCreate,
        store: Annotated[DataStore, StoreDep],
    ) -> SpeciesOut:
        ensure_unique_species(store, payload.name)
        species_id = str(uuid4())
        now = utc_now()
        store.execute(
            """
            INSERT INTO species (
                species_id, name, name_key, description, taxonomic_info,
                reference_images, archived_alias_of, created_at, updated_at, is_archived
            ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, 0)
            """,
            (
                species_id,
                payload.name,
                slug_key(payload.name),
                payload.description,
                payload.taxonomic_info,
                json.dumps(payload.reference_images),
                now,
                now,
            ),
        )
        store.log("create", "species", species_id, payload.model_dump_json())
        row = get_species_or_404(store, species_id)
        return row_to_species(row, 0, 0)

    @router.patch("/species/{species_id}", response_model=SpeciesOut)
    def update_species(
        species_id: str,
        payload: SpeciesUpdate,
        store: Annotated[DataStore, StoreDep],
    ) -> SpeciesOut:
        row = get_species_or_404(store, species_id)
        new_name = payload.name or row["name"]
        if payload.name is not None and payload.name != row["name"]:
            ensure_unique_species(store, payload.name, species_id)
        new_description = (
            row["description"] if payload.description is None else payload.description
        )
        new_taxonomic = (
            row["taxonomic_info"]
            if payload.taxonomic_info is None
            else payload.taxonomic_info
        )
        new_images = (
            json.loads(row["reference_images"])
            if payload.reference_images is None
            else payload.reference_images
        )
        is_archived = int(
            row["is_archived"] if payload.is_archived is None else payload.is_archived
        )
        updated_at = utc_now()
        store.execute(
            """
            UPDATE species
            SET name = ?, name_key = ?, description = ?, taxonomic_info = ?,
                reference_images = ?, updated_at = ?, is_archived = ?
            WHERE species_id = ?
            """,
            (
                new_name,
                slug_key(new_name),
                new_description,
                new_taxonomic,
                json.dumps(new_images),
                updated_at,
                is_archived,
                species_id,
            ),
        )
        if payload.name is not None and payload.name != row["name"]:
            store.execute(
                "UPDATE species SET archived_alias_of = ? WHERE species_id = ?",
                (species_id, species_id),
            )
        store.log("update", "species", species_id, payload.model_dump_json())
        updated = get_species_or_404(store, species_id)
        return row_to_species(updated, *species_counters(store, species_id))

    @router.post("/species/{species_id}/archive", response_model=SpeciesOut)
    def archive_species(
        species_id: str, store: Annotated[DataStore, StoreDep]
    ) -> SpeciesOut:
        row = get_species_or_404(store, species_id)
        store.execute(
            "UPDATE species SET is_archived = 1, updated_at = ? WHERE species_id = ?",
            (utc_now(), species_id),
        )
        store.execute(
            "UPDATE strains SET is_archived = 1 WHERE species_id = ?",
            (species_id,),
        )
        store.execute(
            "UPDATE images SET is_archived = 1 "
            "WHERE strain_id IN "
            "(SELECT strain_id FROM strains WHERE species_id = ?)",
            (species_id,),
        )
        store.log("archive", "species", species_id, row["name"])
        return row_to_species(
            get_species_or_404(store, species_id), *species_counters(store, species_id)
        )

    @router.post("/species/{species_id}/restore", response_model=SpeciesOut)
    def restore_species(
        species_id: str, store: Annotated[DataStore, StoreDep]
    ) -> SpeciesOut:
        store.execute(
            "UPDATE species SET is_archived = 0, updated_at = ? WHERE species_id = ?",
            (utc_now(), species_id),
        )
        store.execute(
            "UPDATE strains SET is_archived = 0 WHERE species_id = ?", (species_id,)
        )
        store.execute(
            "UPDATE images SET is_archived = 0 "
            "WHERE strain_id IN "
            "(SELECT strain_id FROM strains WHERE species_id = ?)",
            (species_id,),
        )
        store.log("restore", "species", species_id, "restored")
        return row_to_species(
            get_species_or_404(store, species_id), *species_counters(store, species_id)
        )

    @router.delete("/species/{species_id}", status_code=204)
    def delete_species(species_id: str, store: Annotated[DataStore, StoreDep]) -> None:
        store.execute(
            "DELETE FROM audit_log WHERE entity_type = 'species' AND entity_id = ?",
            (species_id,),
        )
        store.execute(
            "DELETE FROM images "
            "WHERE strain_id IN "
            "(SELECT strain_id FROM strains WHERE species_id = ?)",
            (species_id,),
        )
        store.execute("DELETE FROM strains WHERE species_id = ?", (species_id,))
        store.execute("DELETE FROM species WHERE species_id = ?", (species_id,))

    @router.get("/strains", response_model=list[StrainOut])
    def list_strains(
        store: Annotated[DataStore, StoreDep],
        species_id: str | None = None,
        include_archived: bool = False,
        search: str | None = None,
    ) -> list[StrainOut]:
        clauses = ["SELECT * FROM strains WHERE (? OR is_archived = 0)"]
        params: list[object] = [include_archived]
        if species_id is not None:
            clauses.append("AND species_id = ?")
            params.append(species_id)
        if search is not None:
            clauses.append("AND name LIKE ?")
            params.append(f"%{search}%")
        rows = store.query(" ".join(clauses), tuple(params))
        result: list[StrainOut] = []
        for row in rows:
            image_count = store.query_one(
                "SELECT COUNT(*) AS c FROM images WHERE strain_id = ?",
                (row["strain_id"],),
            )["c"]
            result.append(
                StrainOut(
                    strain_id=row["strain_id"],
                    name=row["name"],
                    species_id=row["species_id"],
                    created_at=row["created_at"],
                    is_archived=bool(row["is_archived"]),
                    source=row["source"],
                    image_count=int(image_count),
                )
            )
        return result

    @router.post("/strains", response_model=StrainOut, status_code=201)
    def create_strain(
        payload: StrainCreate, store: Annotated[DataStore, StoreDep]
    ) -> StrainOut:
        get_species_or_404(store, payload.species_id)
        strain_id = str(uuid4())
        now = utc_now()
        store.execute(
            "INSERT INTO strains VALUES (?, ?, ?, ?, 0, ?)",
            (strain_id, payload.name, payload.species_id, now, payload.source),
        )
        store.log("create", "strain", strain_id, payload.model_dump_json())
        return StrainOut(
            strain_id=strain_id,
            name=payload.name,
            species_id=payload.species_id,
            created_at=now,
            is_archived=False,
            source=payload.source,
            image_count=0,
        )

    @router.post("/images", response_model=ImageOut, status_code=201)
    def create_image(
        payload: ImageCreate, store: Annotated[DataStore, StoreDep]
    ) -> ImageOut:
        get_strain_or_404(store, payload.strain_id)
        image_id = str(uuid4())
        now = utc_now()
        store.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            (
                image_id,
                payload.strain_id,
                payload.media,
                payload.file_path,
                json.dumps(payload.segments),
                int(payload.indexed_in_qdrant),
                now,
            ),
        )
        store.log("create", "image", image_id, payload.model_dump_json())
        return ImageOut(
            image_id=image_id,
            strain_id=payload.strain_id,
            media=payload.media,
            file_path=payload.file_path,
            segments=payload.segments,
            indexed_in_qdrant=payload.indexed_in_qdrant,
            created_at=now,
            is_archived=False,
        )

    @router.get("/images", response_model=list[ImageOut])
    def list_images(
        store: Annotated[DataStore, StoreDep],
        media: MediaQuery,
        strain_id: str | None = None,
        species_id: str | None = None,
        include_archived: bool = False,
    ) -> list[ImageOut]:
        clauses = [
            "SELECT images.* FROM images "
            "JOIN strains ON strains.strain_id = images.strain_id "
            "WHERE (? OR images.is_archived = 0)"
        ]
        params: list[object] = [include_archived]
        if strain_id is not None:
            clauses.append("AND images.strain_id = ?")
            params.append(strain_id)
        if species_id is not None:
            clauses.append("AND strains.species_id = ?")
            params.append(species_id)
        if media:
            placeholders = ",".join("?" for _ in media)
            clauses.append(f"AND images.media IN ({placeholders})")
            params.extend(media)
        rows = store.query(" ".join(clauses), tuple(params))
        return [
            ImageOut(
                image_id=row["image_id"],
                strain_id=row["strain_id"],
                media=row["media"],
                file_path=row["file_path"],
                segments=json.loads(row["segments"]),
                indexed_in_qdrant=bool(row["indexed_in_qdrant"]),
                created_at=row["created_at"],
                is_archived=bool(row["is_archived"]),
            )
            for row in rows
        ]

    @router.post("/images/batch", response_model=BatchUploadOut, status_code=201)
    def batch_create_images(
        payload: list[BatchImageCreate], store: Annotated[DataStore, StoreDep]
    ) -> BatchUploadOut:
        created: list[ImageOut] = []
        created_strains = 0
        for item in payload:
            get_species_or_404(store, item.species_id)
            strain_id = str(uuid4())
            image_id = str(uuid4())
            now = utc_now()
            store.execute(
                "INSERT INTO strains VALUES (?, ?, ?, ?, 0, ?)",
                (strain_id, item.strain_name, item.species_id, now, item.source),
            )
            store.execute(
                "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    image_id,
                    strain_id,
                    item.media,
                    item.file_path,
                    json.dumps(item.segments),
                    int(item.indexed_in_qdrant),
                    now,
                ),
            )
            store.log("create", "strain", strain_id, item.model_dump_json())
            store.log("create", "image", image_id, item.model_dump_json())
            created_strains += 1
            created.append(
                ImageOut(
                    image_id=image_id,
                    strain_id=strain_id,
                    media=item.media,
                    file_path=item.file_path,
                    segments=item.segments,
                    indexed_in_qdrant=item.indexed_in_qdrant,
                    created_at=now,
                    is_archived=False,
                )
            )
        return BatchUploadOut(
            created_strains=created_strains,
            created_images=len(created),
            images=created,
        )

    @router.delete("/trash", status_code=204)
    def empty_trash(store: Annotated[DataStore, StoreDep]) -> None:
        store.execute("DELETE FROM images WHERE is_archived = 1")
        store.execute("DELETE FROM strains WHERE is_archived = 1")
        store.execute("DELETE FROM species WHERE is_archived = 1")
        store.log("delete", "trash", "all", "empty trash")

    @router.get("/dashboard", response_model=DashboardOut)
    def dashboard(store: Annotated[DataStore, StoreDep]) -> DashboardOut:
        total_images = int(store.query_one("SELECT COUNT(*) AS c FROM images")["c"])
        total_strains = int(store.query_one("SELECT COUNT(*) AS c FROM strains")["c"])
        total_species = int(
            store.query_one("SELECT COUNT(*) AS c FROM species WHERE is_archived = 0")[
                "c"
            ]
        )
        total_media_types = int(
            store.query_one("SELECT COUNT(DISTINCT media) AS c FROM images")["c"]
        )
        learned_count = int(
            store.query_one(
                "SELECT COUNT(*) AS c FROM images "
                "WHERE indexed_in_qdrant = 1 AND is_archived = 0"
            )["c"]
        )
        pending_count = total_images - learned_count
        archived_since_last_training = int(
            store.query_one("SELECT COUNT(*) AS c FROM images WHERE is_archived = 1")[
                "c"
            ]
        )
        species_rows = store.query(
            """
            SELECT species.name, COUNT(images.image_id) AS count
            FROM species
            LEFT JOIN strains ON strains.species_id = species.species_id
            LEFT JOIN images ON images.strain_id = strains.strain_id
                AND images.is_archived = 0
            WHERE species.is_archived = 0
            GROUP BY species.species_id
            ORDER BY count DESC, species.name ASC
            """
        )
        medium_rows = store.query(
            "SELECT media, COUNT(*) AS count FROM images "
            "WHERE is_archived = 0 "
            "GROUP BY media ORDER BY count DESC, media ASC"
        )
        timeline_rows = store.query(
            "SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS count "
            "FROM images GROUP BY day ORDER BY day ASC"
        )
        return DashboardOut(
            total_images=total_images,
            total_strains=total_strains,
            total_species=total_species,
            total_media_types=total_media_types,
            images_per_species=[
                {"name": row["name"], "count": int(row["count"])}
                for row in species_rows
            ],
            images_per_medium=[
                {"name": row["media"], "count": int(row["count"])}
                for row in medium_rows
            ],
            learned_count=learned_count,
            pending_count=pending_count,
            timeline=[
                {"date": row["day"], "count": int(row["count"])}
                for row in timeline_rows
            ],
            archived_since_last_training=archived_since_last_training,
        )

    @router.get("/audit-log", response_model=list[AuditLogOut])
    def audit_log(store: Annotated[DataStore, StoreDep]) -> list[AuditLogOut]:
        rows = store.query("SELECT * FROM audit_log ORDER BY created_at DESC")
        return [AuditLogOut(**dict(row)) for row in rows]

    return router


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name, version="0.1.0", docs_url="/docs", redoc_url="/redoc"
    )
    app.state.store = DataStore()
    app.include_router(make_router())

    @app.get("/health", tags=["health"])
    def healthcheck() -> dict[str, str]:
        return {
            "status": "ok",
            "service": settings.app_name,
            "environment": settings.environment,
        }

    @app.get("/", tags=["meta"])
    def root() -> dict[str, str]:
        return {"name": settings.app_name, "docs": "/docs", "health": "/health"}

    return app


app = create_app()
