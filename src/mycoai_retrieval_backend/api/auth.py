from fastapi import APIRouter

from mycoai_retrieval_backend.schemas.auth import TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login")
async def login() -> TokenResponse:
    return TokenResponse(access_token="development-token", token_type="bearer")


@router.post("/register")
async def register() -> TokenResponse:
    return TokenResponse(access_token="development-token", token_type="bearer")
