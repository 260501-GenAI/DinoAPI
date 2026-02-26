from typing import TypedDict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph

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

# NODES (These still exist! But they won't be part of the agent's decision options)

# Route node (THIS IS THE AGENTIC PART! THE LLM WILL MAKE THE ROUTING DECISION!!)
def agentic_router_node(state:GraphState) -> GraphState:

    # Get the user's query from state
    query = state.get("query", "")

    # Define an agentic prompt for the agentic router
    # Using a different prompting style just to show it
    messages = [
        SystemMessage(content=(
            """
            You are an agentic router that decides which tool to use based on the user's query.
            You may not have to call a tool!!! If a user's query is not related to a tool, that's OK.
            
            You have access to the following tools:
            1. search_dino_docs: Use this tool is the user is asking about people's favorite dinosaurs.
            2. search_plans_docs: Use this tool if the user is asking about upcoming digs or archaeology plans.
            
            If neither tool applies to the query, it's just a general chat - DO NOT CALL A TOOL.
            If you call a tool, call ONLY ONE tool.
            """
        )),
        HumanMessage(content=query)
    ]

    # Invoke the LLM with tools using the prompt
    # The LLM will decide whether to use a tool, and which tool to use
    agentic_response = llm_with_tools.invoke(messages)

    # If there was no tool call, route will equal "chat" for general chats
    if agentic_response.tool_calls == []:
        return {"route":"chat"}

    # If there WAS a tool call, invoke the tool, and store results in the appropriate route
    tool_call = agentic_response.tool_calls[0] # Get the first tool call (there should only be one)
    tool_name = tool_call["name"] # Extracting the name of the tool that was called
    results = TOOL_MAP[tool_name].invoke({"query":query})

    # Automatically set the route to the answer_with_context node and set the docs after the tool is done
    return {
        "route":"answer_with_docs",
        "docs":results
    }

# GENERAL CHAT NODE and ANSWER WITH DOCS NODE will stay largely the same as the other LangGraph service

# Node that uses the stored vectorDB docs to respond to the user
def answer_with_docs(state:GraphState) -> GraphState:

    # Ultimately, this node just invokes the LLM
    # The only difference is it's using the docs stored in state

    # First, extract the query and docs from state
    query = state.get("query", "")
    docs = state.get("docs", [])

    # Set up a prompt - basic, no real personality
    prompt=(
        f"""
        You are a friendly chatbot that takes search results from a VectorDB
        Answer the user's query in a concise but thorough way
        
        Search Results: {docs}
        User Query: {query}
        Answer: 
        """
    )

    # Invoke the LLM! And save the answer in state
    response = llm.invoke(prompt)
    return {"answer":response.text}

# Here's the general chat node that we fall back to if the query isn't related to vector data
def general_chat_node(state:GraphState) -> GraphState:

    # Get the user's query from state
    query = state.get("query", "")

    # Define the prompt
    prompt=(
        f"""
        You are a friendly chatbot that responds to general queries
        Answer the user's query in a concise but thorough way
        
        User Query: {query}
        Answer: 
        """
    )

    # Return the invocation and store it in State
    response = llm.invoke(prompt)
    return {"answer":response.text}


# The Graph Builder (Mostly the same) -----------------
# Uses the new agentic router and the tools are no longer nodes. Also the conditional edge has different outcomes
def build_agentic_graph():

    # First, define the graph builder using the State Graph
    build = StateGraph(GraphState)

    # Register each node
    build.add_node("route", agentic_router_node)
    build.add_node("answer_with_docs", answer_with_docs)
    build.add_node("general_chat", general_chat_node)

    # Set the node that starts the graph (router node in this case)
    build.set_entry_point("route")

    # Set up the branching nodes (conditional edges)
    # A conditional edge is a way to define nodes that MAY run based on a condition
    build.add_conditional_edges(
        "route",
        # This lambda function returns the key we use to choose an edge
        # We're just saying "use the route key" while avoiding writing a whole new function
        lambda state: state["route"],

        # Map that connects the possible "route" values to the appropriate note
        # This is route : node
            # ex: if the route is "chat", invoke the "general_chat_node
        {
            "chat":"general_chat_node",
            "answer_with_docs":"answer_with_docs"
        }
    )

    # Define potential terminal node (stopping points) for the graph
    build.set_finish_point("answer_with_docs")
    build.set_finish_point("general_chat_node")

    # Return the built graph!
    return build.compile()

# Like last time, create a Singleton instance of the graph that we can call in our Router
agentic_graph = build_agentic_graph()