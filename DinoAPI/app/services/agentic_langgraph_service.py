from typing import TypedDict, Any

from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from app.services.vectordb_service import search

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.5
)

# This is the State object for our Graph
# Like in React, State holds data that we want to keep track of
# Each Node in the Graph can read from and write to the State
class GraphState(TypedDict, total=False): #total=False makes all fields optional
    query:str # The user's input to the graph
    route:str # The "routing decision" we make. This tells the app what to invoke next
    docs:list[dict[str, Any]] # Results returned from VectorDB searches
    answer:str # The LLM's answer to the user's query
    #TODO: memory manager field

# =====================(TOOL DEFINITIONS)======================

# The Route node is going to be AGENTIC
# Agents are LLM-powered functions or nodes that can make its own decisions
# Based on the defined tools, our agent can choose which on best fits the user query
# IMPORTANT: each tool needs to be described with """docstrings"""
    # docstrings are how each tool tells the agent what it does and when to use it

@tool(name_or_callable="search_dino_docs")
def search_dino_docs(query:str) -> list[dict[str, Any]]:
    """
    If the user is asking about people's favorite dinosaurs, use this tool
    This tool queries the vectorDB for favorite dino info
    """
    return search("dino_docs", query, k=5)

@tool(name_or_callable="search_plans_docs")
def search_plans_docs(query:str) -> list[dict[str, Any]]:
    """
    If the user is asking about upcoming archaeology plans or plans in general, use this tool
    This tool queries the vectorDB for archaeology plans and dig info
    """
    return search("plans_docs", query, k=5)

# We need some variables that will make the agent aware of the tools

# List of available tools
TOOLS = [search_dino_docs, search_plans_docs]

# Map tool functions to their names
# We need these names so the agentic router (below) can identify what tool to call
TOOL_MAP = {tool.name: tool for tool in TOOLS}

# Get a version of the LLM that's aware of the tools (this is the LLM we'll invoke)
llm_with_tools = llm.bind_tools(TOOLS)