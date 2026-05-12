import json
from collections import defaultdict
from pathlib import Path

from .models import AggregationStrategy, Neighbor, RankedSpecies

SpeciesWeights = dict[str, dict[str, float]]


def load_species_weights(path: str | Path | None) -> SpeciesWeights:
    if path is None or str(path) == "":
        return {}
    weights_path = Path(path)
    if not weights_path.exists():
        return {}
    data = json.loads(weights_path.read_text())
    raw_weights = data.get("weights", data)
    if not isinstance(raw_weights, dict):
        return {}
    weights: SpeciesWeights = {}
    for species, extractor_weights in raw_weights.items():
        if isinstance(species, str) and isinstance(extractor_weights, dict):
            weights[species] = {
                str(extractor): float(weight)
                for extractor, weight in extractor_weights.items()
                if isinstance(weight, int | float)
            }
    return weights


def aggregate_neighbors(
    neighbors: list[Neighbor],
    strategy: AggregationStrategy,
    k: int,
    extractor: str,
    species_weights: SpeciesWeights | None = None,
) -> list[RankedSpecies]:
    scores: defaultdict[str, float] = defaultdict(float)
    counts: defaultdict[str, int] = defaultdict(int)
    weights = species_weights or {}

    for neighbor in neighbors:
        if strategy == "weighted":
            score = neighbor.similarity
        elif strategy == "uni":
            score = 1.0 / k
        elif strategy == "manual_weighted":
            species_weight = weights.get(neighbor.species, {}).get(
                extractor,
                weights.get("default", {}).get(extractor, 1.0),
            )
            score = neighbor.similarity * species_weight
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

        scores[neighbor.species] += score
        counts[neighbor.species] += 1

    ranked = [
        RankedSpecies(species=species, score=score, neighbor_count=counts[species])
        for species, score in scores.items()
    ]
    return sorted(ranked, key=lambda item: (-item.score, item.species))
