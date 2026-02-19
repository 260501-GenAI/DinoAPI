from fastapi import APIRouter
from pydantic import BaseModel

from app.services.vectordb_service import ingest_text

router = APIRouter(
    prefix="/vector",
    tags=["'vector"]
)

# Quick Pydantic Model for ingesting text
class IngestTextRequest(BaseModel):
    text: str

# Endpoint that ingests text
@router.post("/ingest-text")
async def ingest_user_text(input:IngestTextRequest):
    count = ingest_text(f"""{input.text}""")
    return {"ingested chunks: " + count}