from fastapi import APIRouter
from pydantic import BaseModel

from app.services.agentic_langgraph_service import agentic_graph
from app.services.langgraph_service import langgraph

router = APIRouter(
    prefix="/langgraph",
    tags=["langgraph"]
)

# Helper model like we did for langchain and vector ops
class ChatInputModel(BaseModel):
    input:str

# Endpoint that can either:
    # Return a response about fav dinos
    # Return a reponse about boss's dig plans
    # General chat
@router.post("/langgraph")
async def langgraph_chat(chat:ChatInputModel):

    result = langgraph.invoke({"query":chat.input})

    return {
        "route": result.get("route"),
        "response": result.get("answer")
    }

# Same as above, but we're calling the AGENTIC ROUTER now!
# It'll choose which tool to call, then proceed pretty much the same as the old one
@router.post("/agentic-langgraph")
async def agentic_langgraph_chat(chat:ChatInputModel):

    result = agentic_graph.invoke({"query":chat.input})

    return {
        "route": result.get("route"),
        "response": result.get("answer")
    }