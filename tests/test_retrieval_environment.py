from qdrant_client.models import Filter

from mycoai_retrieval_backend.retrieval import build_environment_filter


def test_e1_filter_uses_query_media() -> None:
    query_filter = build_environment_filter("E1", "MEA")

    assert isinstance(query_filter, Filter)
    assert query_filter.must is not None
    assert query_filter.must[0].key == "environment"
    assert query_filter.must[0].match.value == "MEA"


def test_e2_filter_allows_all_media() -> None:
    assert build_environment_filter("E2", None) is None


def test_e3_filter_uses_specific_media() -> None:
    query_filter = build_environment_filter("E3_CYA", "MEA")

    assert isinstance(query_filter, Filter)
    assert query_filter.must is not None
    assert query_filter.must[0].match.value == "CYA"


def test_e4_filter_excludes_specific_media() -> None:
    query_filter = build_environment_filter("E4_CYA", "MEA")

    assert isinstance(query_filter, Filter)
    assert query_filter.must_not is not None
    assert query_filter.must_not[0].match.value == "CYA"
