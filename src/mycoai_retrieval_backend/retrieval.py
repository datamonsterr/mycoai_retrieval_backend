"""Retrieval pipeline: feature extraction, Qdrant search, filtering, aggregation.

Reimplements the fungal-cv-qdrant retrieval flow per the backend-reimplementation
boundary rule.  Inspects src/experiments/retrieval/run.py, src/utils/qdrant_query.py,
and src/lib/cross_validation.py for behaviour contracts but does NOT import from
repos/fungal-cv-qdrant/.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, cast

import cv2
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from skimage.feature import hog


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(features, ord=2)
    if norm == 0:
        return features
    return cast(np.ndarray, features / norm)


def extract_hog_features(
    image: np.ndarray,
    target_size: tuple[int, int] = (128, 128),
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (8, 8),
    cells_per_block: tuple[int, int] = (2, 2),
) -> np.ndarray:
    """Extract Histogram of Oriented Gradients features from a BGR image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, target_size)
    feats = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=False,
        feature_vector=True,
    )
    return _l2_normalize(cast(np.ndarray, feats))


def load_and_extract_features(crop_path: str) -> np.ndarray:
    """Load a crop image from disk and return L2-normalised HOG features."""
    img = cv2.imread(crop_path)
    if img is None or img.size == 0:
        raise ValueError(f"Failed to read image from {crop_path}")
    return extract_hog_features(img)


def get_qdrant_client(qdrant_url: str, qdrant_api_key: str | None) -> QdrantClient:
    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)


def build_filter(
    environment: str | None = None,
    exclude_environment: str | None = None,
) -> Filter | None:
    """Build Qdrant payload filter for environment conditions."""
    conditions: list[Any] = []
    exclude_conditions: list[Any] = []

    if environment is not None:
        conditions.append(
            FieldCondition(key="environment", match=MatchValue(value=environment))
        )
    if exclude_environment is not None:
        exclude_conditions.append(
            FieldCondition(
                key="environment", match=MatchValue(value=exclude_environment)
            )
        )

    if not conditions and not exclude_conditions:
        return None

    if exclude_conditions:
        return Filter(
            must=conditions if conditions else None,
            must_not=exclude_conditions,
        )
    return Filter(must=conditions)


def search_qdrant(
    client: QdrantClient,
    collection_name: str,
    query_vector: np.ndarray,
    feature_type: str,
    k: int,
    environment: str | None = None,
    exclude_environment: str | None = None,
) -> list[dict[str, Any]]:
    """Search Qdrant for nearest neighbours to query_vector."""
    search_filter = build_filter(
        environment=environment, exclude_environment=exclude_environment
    )

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        using=feature_type,
        query_filter=search_filter,
        limit=k,
        with_payload=True,
    )

    neighbours: list[dict[str, Any]] = []
    for point in response.points:
        payload = point.payload or {}
        neighbours.append(
            {
                "image_id": payload.get("image_id"),
                "score": point.score,
                "strain": payload.get("strain"),
                "environment": payload.get("environment"),
                "specy": payload.get("specy"),
                "parent_id": payload.get("parent_id"),
                "segment_index": payload.get("segment_index"),
            }
        )
    return neighbours


def filter_siblings(
    neighbours: list[dict[str, Any]], query_parent_id: str
) -> list[dict[str, Any]]:
    """Remove neighbours that belong to the same parent image as the query."""
    return [n for n in neighbours if n.get("parent_id") != query_parent_id]


def aggregate_predictions(
    segment_results: list[dict[str, Any]],
    strategy: str = "weighted",
    strain_to_specy: dict[str, str] | None = None,
) -> list[tuple[str, float]]:
    """Aggregate per-segment neighbour results into species ranking.

    Args:
        segment_results: List of dicts each containing a "neighbours" key.
        strategy: "weighted" (cosine-similarity-weighted) or "uni" (uniform).
        strain_to_specy: Optional mapping from strain→species for unknown species.

    Returns:
        Sorted list of (species, score) tuples, highest first.
    """
    species_scores: Counter[str] = Counter()
    species_counts: Counter[str] = Counter()

    for result in segment_results:
        for neighbour in result["neighbours"]:
            specy: Any = neighbour.get("specy")
            score = neighbour.get("score", 0.0)

            if not specy or specy == "unknown":
                strain = neighbour.get("strain")
                if strain and strain_to_specy:
                    specy = strain_to_specy.get(strain, "unknown")

            if specy and specy != "unknown":
                species_scores[specy] += score
                species_counts[specy] += 1

    total_neighbours = sum(species_counts.values())
    aggregated: list[tuple[str, float]] = []

    for specy, total_score in species_scores.items():
        if strategy == "weighted":
            final_score = (
                total_score / total_neighbours if total_neighbours > 0 else 0.0
            )
        elif strategy == "uni":
            count = species_counts[specy]
            final_score = count / total_neighbours if total_neighbours > 0 else 0.0
        else:
            final_score = float(total_score)
        aggregated.append((specy, final_score))

    aggregated.sort(key=lambda x: x[1], reverse=True)
    return aggregated


def resolve_environment(
    strategy: str,
    image_media: str,
    e3_medium: str | None = None,
    e4_exclude_medium: str | None = None,
) -> tuple[str | None, str | None]:
    """Return (include_environment, exclude_environment) for the given strategy.

    E1: same medium — only match images from the same medium.
    E2: all media — no filter.
    E3: specific medium — only match a user-chosen medium.
    E4: exclude medium — exclude a user-chosen medium.
    """
    if strategy == "E1":
        return (image_media, None)
    if strategy == "E2":
        return (None, None)
    if strategy == "E3":
        return (e3_medium, None)
    if strategy == "E4":
        return (None, e4_exclude_medium)
    return (None, None)


def retrieve(
    client: QdrantClient,
    collection_name: str,
    strain: str,
    images: list[dict[str, Any]],
    k: int = 5,
    aggregation: str = "weighted",
    environment_strategy: str = "E1",
    e3_medium: str | None = None,
    e4_exclude_medium: str | None = None,
    feature_type: str = "hog",
) -> dict[str, Any]:
    """Run the full retrieval pipeline for one strain.

    Args:
        client: Qdrant client.
        collection_name: Target collection name.
        strain: Strain identifier.
        images: List of image dicts with keys: image_id, media, segments.
        k: Number of neighbours per segment.
        aggregation: "weighted" or "uni".
        environment_strategy: "E1" | "E2" | "E3" | "E4".
        e3_medium: Required medium when strategy=E3.
        e4_exclude_medium: Excluded medium when strategy=E4.
        feature_type: Vector name in Qdrant ("hog" default).

    Returns:
        Dict with keys: strain, rankings, query_details.
    """
    segment_results: list[dict[str, Any]] = []
    total_neighbours = 0

    for image in images:
        image_id = image["image_id"]
        media = image["media"]
        segments = image.get("segments", [])

        include_env, exclude_env = resolve_environment(
            environment_strategy, media, e3_medium, e4_exclude_medium
        )

        for seg in segments:
            crop_path = seg["crop_path"]
            features = load_and_extract_features(crop_path)

            neighbours = search_qdrant(
                client=client,
                collection_name=collection_name,
                query_vector=features,
                feature_type=feature_type,
                k=k,
                environment=include_env,
                exclude_environment=exclude_env,
            )

            neighbours = filter_siblings(neighbours, image_id)

            total_neighbours += len(neighbours)
            segment_results.append(
                {
                    "image_id": image_id,
                    "segment_index": seg["segment_index"],
                    "neighbours": neighbours,
                }
            )

    rankings_flat = aggregate_predictions(segment_results, strategy=aggregation)

    rankings = [
        {"rank": i + 1, "species": spec, "score": round(score, 4)}
        for i, (spec, score) in enumerate(rankings_flat)
    ]
    if not rankings:
        rankings = [{"rank": 1, "species": "unknown", "score": 0.0}]

    return {
        "strain": strain,
        "rankings": rankings,
        "query_details": {
            "k": k,
            "aggregation": aggregation,
            "environment_strategy": environment_strategy,
            "total_neighbors_queried": total_neighbours,
        },
    }
