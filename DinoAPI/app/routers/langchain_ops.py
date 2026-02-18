from fastapi import APIRouter
from pydantic import BaseModel

from app.services.langchain_service import get_basic_chain, get_sequential_chain

# Same old router setup
router = APIRouter(
    prefix="/langchain",
    tags=["langchain"]
)

# I'm going to make a quick Pydantic model that will represent the user's input
# This helps it play nice with FastAPI
class ChatInputModel(BaseModel):
    input:str

# Import the chains we defined in the Service for use in the endpoints below
basic_chain = get_basic_chain()
refined_answer_chain = get_sequential_chain()

# General chat endpoint with no memory or any other fancy features
@router.post("/chat")
async def general_chat(chat:ChatInputModel):
    # Now we just invoke the chain with the user's input!
    return basic_chain.invoke(input={"input":chat.input})

# This endpoint is for the more professional chat using our sequential chain
@router.post("/refined-chat")
async def refined_chat(chat:ChatInputModel):
    return refined_answer_chain.invoke(input={"input":chat.input})
