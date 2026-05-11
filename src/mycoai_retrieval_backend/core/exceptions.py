from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ProblemDetail(BaseModel):
    type: str = "about:blank"
    title: str
    status: int
    detail: str
    instance: str


class AppError(Exception):
    def __init__(self, title: str, status_code: int, detail: str) -> None:
        self.title = title
        self.status_code = status_code
        self.detail = detail


def problem_response(request: Request, error: AppError) -> JSONResponse:
    problem = ProblemDetail(
        title=error.title,
        status=error.status_code,
        detail=error.detail,
        instance=str(request.url.path),
    )
    return JSONResponse(status_code=error.status_code, content=problem.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, error: AppError) -> JSONResponse:
        return problem_response(request, error)

    @app.exception_handler(404)
    async def not_found_handler(request: Request, error: Exception) -> JSONResponse:
        return problem_response(
            request,
            AppError(
                title=HTTPStatus.NOT_FOUND.phrase,
                status_code=HTTPStatus.NOT_FOUND.value,
                detail="Resource not found",
            ),
        )
