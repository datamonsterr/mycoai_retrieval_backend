from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, cast
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import Filter

from .aggregation import SpeciesWeights, aggregate_neighbors
from .environment import build_environment_filter
from .models import (
    AggregationStrategy,
    BatchRetrievalRequest,
    BatchRetrievalResponse,
    BatchStrainResult,
    Neighbor,
    RetrievalRequest,
    RetrievalResponse,
    SegmentQueryResult,
    StrainRetrievalRequest,
)


class SegmentStore(Protocol):
    def get_segments(self, image_id: str) -> list[dict[str, object]]: ...


@dataclass(frozen=True)
class SegmentRecord:
    segment_id: str
    image_id: str
    parent_item_id: str | None
    environment: str | None
    vector: list[float]
    metadata: dict[str, object]


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _float_sequence(value: object) -> list[float]:
    if isinstance(value, list | tuple) and all(
        isinstance(x, int | float) for x in value
    ):
        return [float(x) for x in value]
    raise ValueError("segment vector must be a list of numbers")


class QdrantSegmentStore:
    def __init__(self, client: QdrantClient, collection_name: str, feature_name: str):
        self.client = client
        self.collection_name = collection_name
        self.feature_name = feature_name

    def get_segments(self, image_id: str) -> list[dict[str, object]]:
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=build_image_filter(image_id),
            limit=10,
            with_payload=True,
            with_vectors=True,
        )
        segments: list[dict[str, object]] = []
        for point in points:
            payload = point.payload or {}
            vector_data = point.vector
            if (
                not isinstance(vector_data, Mapping)
                or self.feature_name not in vector_data
            ):
                continue
            segments.append(
                {
                    "segment_id": str(
                        payload.get("segment_id") or payload.get("image_id") or point.id
                    ),
                    "image_id": str(payload.get("image_id") or image_id),
                    "parent_item_id": payload.get("parent_item_id")
                    or payload.get("parent_id"),
                    "environment": payload.get("environment"),
                    "vector": vector_data[self.feature_name],
                    "metadata": dict(payload),
                }
            )
        return segments[:3]


def build_image_filter(image_id: str) -> Filter:
    from qdrant_client.models import FieldCondition, MatchValue

    return Filter(
        must=[FieldCondition(key="image_id", match=MatchValue(value=image_id))]
    )


class RetrievalPipeline:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        default_feature_extractor: str,
        species_weights: SpeciesWeights | None = None,
        segment_store: SegmentStore | None = None,
    ):
        self.client = client
        self.collection_name = collection_name
        self.default_feature_extractor = default_feature_extractor
        self.species_weights = species_weights or {}
        self.segment_store = segment_store

    def query_image(self, request: RetrievalRequest) -> RetrievalResponse:
        return self._query_images(
            image_ids=[request.image_id],
            k=request.k,
            env_strategy=request.env_strategy,
            aggregation=request.aggregation,
            feature_extractor=(
                request.feature_extractor or self.default_feature_extractor
            ),
        )

    def query_strain(self, request: StrainRetrievalRequest) -> RetrievalResponse:
        return self._query_images(
            image_ids=request.image_ids,
            k=request.k,
            env_strategy=request.env_strategy,
            aggregation=request.aggregation,
            feature_extractor=(
                request.feature_extractor or self.default_feature_extractor
            ),
        )

    def query_batch(self, request: BatchRetrievalRequest) -> BatchRetrievalResponse:
        started = perf_counter()
        results: list[BatchStrainResult] = []
        for strain in request.strains:
            try:
                results.append(
                    BatchStrainResult(
                        strain_id=strain.strain_id,
                        result=self.query_strain(strain),
                    )
                )
            except Exception as exc:
                results.append(
                    BatchStrainResult(strain_id=strain.strain_id, error=str(exc))
                )
        return BatchRetrievalResponse(
            results=results,
            metrics={
                "response_time_ms": round((perf_counter() - started) * 1000.0, 3),
                "strain_count": len(request.strains),
                "success_count": sum(1 for result in results if result.error is None),
            },
        )

    def _query_images(
        self,
        image_ids: list[str],
        k: int,
        env_strategy: str,
        aggregation: AggregationStrategy,
        feature_extractor: str,
    ) -> RetrievalResponse:
        started = perf_counter()
        segment_results: list[SegmentQueryResult] = []
        errors: list[str] = []

        for image_id in image_ids:
            try:
                segments = self._load_segments(image_id, feature_extractor)
            except Exception as exc:
                errors.append(f"{image_id}: {exc}")
                continue

            if not segments:
                errors.append(f"{image_id}: no segments found")
                continue

            for segment in segments:
                try:
                    neighbors = self._query_segment(
                        segment=segment,
                        k=k,
                        env_strategy=env_strategy,
                        feature_extractor=feature_extractor,
                    )
                    segment_results.append(
                        SegmentQueryResult(
                            segment_id=segment.segment_id,
                            image_id=image_id,
                            environment=segment.environment,
                            neighbors=neighbors,
                        )
                    )
                except Exception as exc:
                    message = f"{segment.segment_id}: {exc}"
                    errors.append(message)
                    segment_results.append(
                        SegmentQueryResult(
                            segment_id=segment.segment_id,
                            image_id=image_id,
                            environment=segment.environment,
                            error=str(exc),
                        )
                    )

        all_neighbors = [
            neighbor for result in segment_results for neighbor in result.neighbors
        ]
        predictions = aggregate_neighbors(
            all_neighbors,
            strategy=aggregation,
            k=k,
            extractor=feature_extractor,
            species_weights=self.species_weights,
        )
        if segment_results and all(result.error for result in segment_results):
            errors.append("all segments failed")

        return RetrievalResponse(
            request_id=str(uuid4()),
            image_ids=image_ids,
            predictions=predictions,
            segments=segment_results,
            aggregation=aggregation,
            feature_extractor=feature_extractor,
            errors=errors,
            metrics={
                "response_time_ms": round((perf_counter() - started) * 1000.0, 3),
                "neighbors_retrieved": len(all_neighbors),
                "segment_count": len(segment_results),
                "success": int(
                    bool(predictions)
                    and not all(result.error for result in segment_results)
                ),
            },
        )

    def _load_segments(
        self, image_id: str, feature_extractor: str
    ) -> list[SegmentRecord]:
        store = self.segment_store or QdrantSegmentStore(
            self.client,
            self.collection_name,
            feature_extractor,
        )
        records = []
        for raw in store.get_segments(image_id):
            records.append(
                SegmentRecord(
                    segment_id=str(raw["segment_id"]),
                    image_id=str(raw["image_id"]),
                    parent_item_id=_string_or_none(raw.get("parent_item_id")),
                    environment=_string_or_none(raw.get("environment")),
                    vector=_float_sequence(raw.get("vector")),
                    metadata=cast(dict[str, object], raw.get("metadata", {})),
                )
            )
        return records[:3]

    def _query_segment(
        self,
        segment: SegmentRecord,
        k: int,
        env_strategy: str,
        feature_extractor: str,
    ) -> list[Neighbor]:
        query_filter = build_environment_filter(env_strategy, segment.environment)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=segment.vector,
            using=feature_extractor,
            query_filter=query_filter,
            limit=k * 10,
            with_payload=True,
        )
        neighbors: list[Neighbor] = []
        for point in response.points:
            payload = point.payload or {}
            parent_item_id = _string_or_none(
                payload.get("parent_item_id") or payload.get("parent_id")
            )
            if (
                segment.parent_item_id is not None
                and parent_item_id == segment.parent_item_id
            ):
                continue
            species = payload.get("species") or payload.get("specy")
            if not isinstance(species, str) or species == "unknown":
                continue
            neighbors.append(
                Neighbor(
                    image_id=str(payload.get("image_id") or point.id),
                    segment_id=_string_or_none(payload.get("segment_id")),
                    parent_item_id=parent_item_id,
                    species=species,
                    strain=_string_or_none(payload.get("strain")),
                    environment=_string_or_none(payload.get("environment")),
                    similarity=float(point.score),
                    metadata=dict(payload),
                )
            )
            if len(neighbors) >= k:
                break
        return neighbors
