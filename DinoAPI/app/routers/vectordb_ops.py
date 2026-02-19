from fastapi import APIRouter
from pydantic import BaseModel

from app.services.vectordb_service import ingest_text, search_dino_store

router = APIRouter(
    prefix="/vector",
    tags=["'vector"]
)

# Quick Pydantic Model for ingesting text
class IngestTextRequest(BaseModel):
    text: str

# Another quick model for similarity search requests
class SearchRequest(BaseModel):
    query: str
    k:int = 6

# Endpoint that ingests text
@router.post("/ingest-text")
async def ingest_user_text(input:IngestTextRequest):
    count = ingest_text(f"""{input.text}""")
    return {f"ingested chunks: {count}"}

# Endpoint that does a similarity based on a user's query
@router.post("/search")
async def similarity_search(request:SearchRequest):
    return search_dino_store(request.query, request.k)