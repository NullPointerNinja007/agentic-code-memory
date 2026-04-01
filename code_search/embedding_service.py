"""
Embedding service module for generating compact code descriptions and indexing them in ChromaDB.

This module provides functions to:
1. Generate highly specific, space-efficient descriptions of code snippets using GPT-5.1
2. Index code snippets with embeddings in a persistent ChromaDB vector store
"""

import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables
load_dotenv()

# Initialize OpenAI client
_api_key = os.getenv("OPENAI_API_KEY")
# Strip quotes and whitespace in case the .env file has quotes around the value
if _api_key:
    _api_key = _api_key.strip().strip('"').strip("'")
if not _api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable is not set. "
        "Please create a .env file with OPENAI_API_KEY=your_key"
    )

_client = OpenAI(api_key=_api_key)

# ChromaDB client configuration
# Supports both local (PersistentClient) and remote (HttpClient) modes
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CHROMA_DB_PATH = str(_PROJECT_ROOT / "chroma_db")
_CHROMA_SERVER_HOST = os.getenv("CHROMA_SERVER_HOST")  # e.g., "localhost" or "your-chromadb-server.com"
_CHROMA_SERVER_PORT = os.getenv("CHROMA_SERVER_PORT", "8000")
# Use HttpClient if CHROMA_SERVER_HOST is set (for serverless/production)
# Otherwise use PersistentClient (for local development)
if _CHROMA_SERVER_HOST:
    _chroma_client = chromadb.HttpClient(
        host=_CHROMA_SERVER_HOST,
        port=int(_CHROMA_SERVER_PORT)
    )
else:
    _chroma_client = chromadb.PersistentClient(path=_CHROMA_DB_PATH)


def generate_compact_code_description(
    code: str,
    user_description: Optional[str] = None,
) -> str:
    """
    Generate a highly specific, space-efficient description string for a code snippet using GPT-5.1.
    
    The description is designed to maximize semantic information for embeddings while being
    extremely compact. It follows a semicolon-separated, tagged format.
    
    Args:
        code: The code snippet to describe (as a string)
        user_description: Optional user-provided description to guide the analysis
        
    Returns:
        A single-line, compact description string with tagged fields separated by semicolons
        
    Raises:
        Exception: If the OpenAI API call fails
    """
    system_prompt = """You are a code analysis expert. Your task is to generate a highly specific, 
space-efficient description string for code snippets. 

The description must be:
- Extremely specific and dense
- A single line of text only
- Semicolon-separated, tagged format
- Designed to maximize semantic information for embeddings
- NO JSON, NO bullets, NO natural language paragraphs
- NO explanations, quotes, backticks, or surrounding text

Output format example:
LANG:python; KIND:http-handler; LIBS:fastapi,sqlalchemy; PURPOSE:create-user; INPUTS:user_data(json,validated); OUTPUT:user_id; SIDE_EFFECTS:db_insert(users); ERROR:400(validation),409(duplicate); ALGO:none;

Required fields (include all relevant ones):
- LANG: programming language
- KIND: code type (http-handler, class, function, script, utility, etc.)
- LIBS: major libraries/frameworks used (comma-separated)
- PURPOSE: main purpose/role of the code
- INPUTS: expected inputs with types/roles (comma-separated, use parentheses for details)
- OUTPUT: expected outputs/return values
- SIDE_EFFECTS: side effects like db_insert, network_call, file_io, etc.
- ERROR: error handling patterns (status codes, exception types)
- ALGO: high-level algorithmic category (sorting, hashing, graph-traversal, dynamic-programming, etc.)
- COMPLEXITY: time/space complexity if obvious (e.g., O(n_log_n), O(n_squared))

Remember: Output ONLY the single compact description line. Nothing else."""

    user_prompt_parts = [
        "Analyze the following code snippet and generate a compact description string:",
        "",
        "```",
        code,
        "```",
    ]
    
    if user_description:
        user_prompt_parts.insert(1, f"Additional context: {user_description}")
    
    user_prompt = "\n".join(user_prompt_parts)
    
    try:
        response = _client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_completion_tokens=500,
        )
        
        description = response.choices[0].message.content.strip()
        
        # Remove any quotes, backticks, or markdown code blocks that might have been included
        description = description.strip('`"\'\n').strip()
        
        return description
        
    except Exception as e:
        raise Exception(f"Failed to generate code description: {str(e)}") from e


def index_snippet_in_chroma(
    uuid: str,
    code: str,
    user_description: Optional[str] = None,
) -> str:
    """
    Generate a compact description for a code snippet, create an embedding, and index it in ChromaDB.
    
    This function:
    1. Generates a compact description using GPT-5.1
    2. Creates an embedding vector using text-embedding-3-small
    3. Upserts the embedding into ChromaDB with the UUID as the ID
    
    Args:
        uuid: Unique identifier for the code snippet (will be used as ChromaDB document ID)
        code: The code snippet to index (as a string)
        user_description: Optional user-provided description to guide the analysis
        
    Returns:
        The compact description string (so callers can optionally store it elsewhere)
        
    Raises:
        Exception: If description generation or embedding creation fails
        ValueError: If uuid is empty
    """
    if not uuid or not uuid.strip():
        raise ValueError("uuid must be a non-empty string")
    
    # Generate compact description
    description = generate_compact_code_description(code, user_description)
    
    # Get or create collection
    collection = _chroma_client.get_or_create_collection(
        name="snippets",
        metadata={"description": "Code snippets with compact descriptions"}
    )
    
    # Generate embedding
    try:
        embedding_response = _client.embeddings.create(
            model="text-embedding-3-small",
            input=description,
        )
        embedding = embedding_response.data[0].embedding
        
    except Exception as e:
        raise Exception(f"Failed to create embedding: {str(e)}") from e
    
    # Upsert into ChromaDB
    try:
        collection.upsert(
            ids=[uuid],
            embeddings=[embedding],
            metadatas=[{
                "uuid": uuid,
                "compact_description": description,
            }],
        )
    except Exception as e:
        raise Exception(f"Failed to upsert into ChromaDB: {str(e)}") from e
    
    return description


def search_similar_snippets(
    query: str,
    k: int = 1,
) -> list[str]:
    """
    Search for similar code snippets in ChromaDB using a text query.
    
    This function:
    1. Generates an embedding for the query string using text-embedding-3-small
    2. Performs similarity search in ChromaDB
    3. Returns the UUIDs of the top-k most similar snippets
    
    Args:
        query: Text query string to search for
        k: Number of top results to return (default: 1)
        
    Returns:
        List of UUIDs ranked by similarity (most similar first).
        If there are fewer than k items in the database, returns all available UUIDs.
        
    Raises:
        Exception: If embedding generation or database query fails
        ValueError: If query is empty or k is less than 1
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    if k < 1:
        raise ValueError("k must be at least 1")
    
    # Get the collection
    try:
        collection = _chroma_client.get_or_create_collection(
            name="snippets",
            metadata={"description": "Code snippets with compact descriptions"}
        )
    except Exception as e:
        raise Exception(f"Failed to get ChromaDB collection: {str(e)}") from e
    
    # Get the current count of items in the collection
    collection_count = collection.count()
    
    # If no items in collection, return empty list
    if collection_count == 0:
        return []
    
    # Adjust k to not exceed the number of items in the collection
    actual_k = min(k, collection_count)
    
    # Generate embedding for the query
    try:
        embedding_response = _client.embeddings.create(
            model="text-embedding-3-small",
            input=query,
        )
        query_embedding = embedding_response.data[0].embedding
    except Exception as e:
        raise Exception(f"Failed to create query embedding: {str(e)}") from e
    
    # Perform similarity search
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
        )
    except Exception as e:
        raise Exception(f"Failed to query ChromaDB: {str(e)}") from e
    
    # Extract UUIDs from results (results['ids'] is a list of lists)
    if results['ids'] and len(results['ids']) > 0:
        uuids = results['ids'][0]  # First (and only) query result
        return uuids
    else:
        return []


def clear_vector_database() -> None:
    """
    Clear all entries from the ChromaDB vector database.
    
    This function:
    1. Deletes the existing 'snippets' collection
    2. Recreates an empty 'snippets' collection ready for new entries
    
    After calling this function, the database will be empty but ready to accept
    new entries via index_snippet_in_chroma().
    
    Raises:
        Exception: If the collection deletion or recreation fails
    """
    try:
        # Try to delete the collection if it exists
        try:
            _chroma_client.delete_collection(name="snippets")
        except Exception:
            # Collection might not exist, which is fine
            pass
        
        # Recreate an empty collection ready for new entries
        _chroma_client.get_or_create_collection(
            name="snippets",
            metadata={"description": "Code snippets with compact descriptions"}
        )
    except Exception as e:
        raise Exception(f"Failed to clear vector database: {str(e)}") from e
