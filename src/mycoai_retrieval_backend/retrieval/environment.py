from qdrant_client.models import FieldCondition, Filter, MatchValue


def build_environment_filter(strategy: str, query_media: str | None) -> Filter | None:
    if strategy == "E1":
        if query_media is None:
            raise ValueError("query_media is required for E1 environment strategy")
        return Filter(
            must=[
                FieldCondition(
                    key="environment",
                    match=MatchValue(value=query_media),
                )
            ]
        )
    if strategy == "E2":
        return None
    if strategy.startswith("E3_"):
        env = strategy[3:]
        if not env:
            raise ValueError("E3 strategy requires an environment suffix")
        return Filter(
            must=[FieldCondition(key="environment", match=MatchValue(value=env))]
        )
    if strategy.startswith("E4_"):
        env = strategy[3:]
        if not env:
            raise ValueError("E4 strategy requires an environment suffix")
        return Filter(
            must_not=[FieldCondition(key="environment", match=MatchValue(value=env))]
        )
    raise ValueError(f"Unknown environment strategy: {strategy}")
