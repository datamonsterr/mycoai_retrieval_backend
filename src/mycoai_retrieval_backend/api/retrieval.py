from fastapi import APIRouter

from mycoai_retrieval_backend.schemas.retrieval import RetrievalQuery, RetrievalResponse

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.post("/query")
async def query(payload: RetrievalQuery) -> RetrievalResponse:
    return RetrievalResponse(query=payload.query, results=[])
