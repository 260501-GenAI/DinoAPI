from fastapi import APIRouter
from pydantic import BaseModel

from app.services.langchain_service import get_basic_chain
from app.services.vectordb_service import ingest_text, search

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

# Last quick model for LLM queries
class ChatInputModel(BaseModel):
    input:str

# Import the basic chain from the langchain service
basic_chain = get_basic_chain()

# Endpoint that ingests text
# User will pass in "dino_docs" or "plans_docs" depending on the collection they need
    # (Realistically, the front end would automatically supply the collection to use)
@router.post("/ingest-text")
async def ingest_user_text(collection:str, input:IngestTextRequest):
    count = ingest_text(collection, input.text)
    return {f"ingested chunks: {count}"}

# Endpoint that does a similarity based on a user's query
@router.post("/search")
async def similarity_search(collection:str, request:SearchRequest):
    return search(collection, request.query, request.k)


# Endpoint for querying the LLM about the dino docs (general chat-ish)
@router.post("/dino-doc-rag")
async def dino_doc_rag(chat:ChatInputModel):
    # Extract results from the VectorDB
    results = search("dino_docs", chat.input, k=5)

    # Quick prompt that tells the LLM the results of the search
    # and asks it to respond to the user's query using those results
    prompt = f"""
    
    Based on the following extracted info about user's favorite dinos,
    Answer the user's query as best you can, using ONLY the extracted info
    If there's no relevant info, you can say that
    
    Extracted Info: {results}
    User Query: {chat.input}

    """

    # Invoke the chain with the prompt and return the response
    return basic_chain.invoke(input={"input": prompt})


# Endpoint for querying the LLM about archeology plans (a bit more formal)
# TODO: we never actually changed the tone of the prompt cuz I ran out of time
@router.post("/plans-doc-rag")
async def plans_doc_rag(chat:ChatInputModel):
    # Extract results from the VectorDB
    results = search("plans_docs", chat.input, k=5)

    # Quick prompt that tells the LLM the results of the search
    # and asks it to respond to the user's query using those results
    prompt = f"""
    
    Based on the following extracted info about upcoming archaeology plans,
    Answer the user's query as best you can, using ONLY the extracted info
    If there's no relevant info, you can say that
    
    Extracted Info: {results}
    User Query: {chat.input}

    """

    # Invoke the chain with the prompt and return the response
    return basic_chain.invoke(input={"input": prompt})