from mycoai_retrieval_backend.retrieval import Neighbor, aggregate_neighbors


def test_weighted_aggregation_sums_similarity_by_species() -> None:
    ranked = aggregate_neighbors(
        [
            Neighbor(image_id="n1", species="Species A", similarity=0.9),
            Neighbor(image_id="n2", species="Species B", similarity=0.8),
            Neighbor(image_id="n3", species="Species A", similarity=0.7),
        ],
        strategy="weighted",
        k=5,
        extractor="EfficientNetV2B0",
    )

    assert [item.species for item in ranked] == ["Species A", "Species B"]
    assert ranked[0].score == 1.6
    assert ranked[0].neighbor_count == 2


def test_uniform_aggregation_adds_one_over_k_per_neighbor() -> None:
    ranked = aggregate_neighbors(
        [
            Neighbor(image_id="n1", species="Species A", similarity=0.1),
            Neighbor(image_id="n2", species="Species A", similarity=0.9),
            Neighbor(image_id="n3", species="Species B", similarity=0.99),
        ],
        strategy="uni",
        k=5,
        extractor="EfficientNetV2B0",
    )

    assert ranked[0].species == "Species A"
    assert ranked[0].score == 0.4
    assert ranked[1].score == 0.2


def test_manual_weighted_aggregation_uses_species_weight_then_default() -> None:
    ranked = aggregate_neighbors(
        [
            Neighbor(image_id="n1", species="Species A", similarity=1.0),
            Neighbor(image_id="n2", species="Species B", similarity=1.0),
        ],
        strategy="manual_weighted",
        k=5,
        extractor="EfficientNetV2B0",
        species_weights={
            "Species A": {"EfficientNetV2B0": 2.0},
            "default": {"EfficientNetV2B0": 0.5},
        },
    )

    assert [(item.species, item.score) for item in ranked] == [
        ("Species A", 2.0),
        ("Species B", 0.5),
    ]
