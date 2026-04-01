import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CHROMA_DIR = _PROJECT_ROOT / "chroma_db"

load_dotenv()

_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise ValueError("OPENAI_API_KEY is not set")

_client = OpenAI(api_key=_api_key)

_chroma_client = chromadb.PersistentClient(
    path=str(_CHROMA_DIR), settings=Settings(allow_reset=True)
)
_collection = _chroma_client.get_or_create_collection("snippets")


def generate_embedding(description: str) -> list[float]:
    """
    Generate an embedding for a description.
    """
    _model = "text-embedding-3-large"

    try:
        response = _client.embeddings.create(
            model=_model,
            input=description,
        )
        return response.data[0].embedding
    except Exception:
        raise Exception("Failed to generate embedding")


def generate_code_description(
    code: str, user_description: Optional[str] = None
) -> str:
    """
    Generate a description for a code snippet.
    """
    system_prompt = """You are a code analysis expert. Your task is to generate a description for a code snippet.
    The description (User Description) written by the user (author of the code) and the code (Code) is provided to you. Using those generate a description that must be:
     
    - Extremely specific and dense
    - A single line of continous text without any newlines or spaces
    - Semicolon-separated, tagged format
    - Designed to maximize semantic information for embeddings
    - NO JSON, NO bullets, NO natural language paragraphs
    - NO explanations, quotes, backticks, or surrounding text
    - Inlcude technical tags which will help in similarity search.
    """
    user_prompt = f"Code: {code}\nUser Description: {user_description}"
    try:
        response = _client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_completion_tokens=500,
        )
        return response.choices[0].message.content
    except Exception:
        raise Exception("Failed to generate code description")


def addDB(code: str, description: str, UUID: str) -> None:
    """
    Add a code snippet to the database.
    """
    _description = generate_code_description(code, description)
    _embedding = generate_embedding(_description)

    try:
        _collection.upsert(
            ids=[UUID],
            embeddings=[_embedding],
            metadatas=[{"code": code, "description": _description}],
        )
    except Exception:
        raise Exception("Failed to add DB entry")


def reset_vector_database() -> bool:
    """
    Reset the ChromaDB vector database by deleting the collection and creating a new one.
    """
    _chroma_client.reset()
    return True


def delete_DB_entry(UUID: str) -> None:
    """
    Delete a code snippet from the database.
    """
    try:
        _collection.delete(ids=[UUID])
    except Exception:
        raise Exception("Failed to delete DB entry")


def searchDB(query: str, top_k: int = 1) -> list[str]:
    """
    Search the database for code snippets.
    """
    try:
        _embedding = generate_embedding(query)
        results = _collection.query(
            query_embeddings=[_embedding],
            n_results=top_k,
            include=["metadatas"],
        )
        return [item["code"] for item in results["metadatas"][0]]
    except Exception:
        raise Exception("Failed to search DB")


def get_number_of_DB_entries() -> int:
    """
    Get the number of code snippets in the database.
    """
    try:
        return _collection.count()
    except Exception:
        raise Exception("Failed to get number of DB entries")
