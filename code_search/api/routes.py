import uuid
from typing import Optional

from fastapi import APIRouter, status
from pydantic import BaseModel

from code_search.handle_db import addDB, searchDB

router = APIRouter()


class CodeRequest(BaseModel):
    code: str
    user_description: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 1


@router.post("/add_code", status_code=status.HTTP_201_CREATED)
async def add_code(code_request: CodeRequest) -> None:
    """
    Add a code snippet to the database.
    """
    UUID = str(uuid.uuid4())
    addDB(code_request.code, code_request.user_description, UUID)
    return None


@router.post("/search_code", status_code=status.HTTP_200_OK)
async def search_code(search_request: SearchRequest) -> list[str]:
    """
    Search the database for code snippets.
    """
    return searchDB(search_request.query, search_request.top_k)
