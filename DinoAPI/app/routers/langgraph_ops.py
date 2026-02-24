from fastapi import APIRouter
from pydantic import BaseModel

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
    # TODO: general chat
@router.post("/langgraph")
async def langgraph_chat(chat:ChatInputModel):
    result = langgraph.invoke({"query":chat.input})

    return result