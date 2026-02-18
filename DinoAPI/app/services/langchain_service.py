# This service will store different chains that help us query our LLM
# A chain is sequence of actions that we can send to the LLM in one go.
# LangCHAIN is all about building CHAINS that help us get good responses from the LLM
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Define the LLM we're going to use (llama3.2:3b which we installed locally)
llm = ChatOllama(
    model="llama3.2:3b", # The model we're using
    temperature=0.5 # Temp goes from 0-1. Higher temp = more creative responses from the LLM
)

# Define the prompt we'll send to the LLM to define tone, context, and instructions
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a helpful chatbot that answer questions about dinosaurs, paleontology, 
    and general prehistoric queries. 
    
    You speak like an old crazy prospector who really loves fossils and dinosaurs.
    You are kind and helpful, but tend to go off the rails and ramble a little bit. 
    
    You never answer questions that don't have to do with dinosaurs or prehistory.
    You don't provide further suggestions beyond what the user asks."""),
    ("user", "{input}")
])

# Our first basic chain - just combines the prompt and the LLM,
# Returning something we can query!
def get_basic_chain():
    # This basic chain was defined using LCEL (LangChain Expression Language)
    # The components in it are just the llm and prompt we defined above
    chain = prompt | llm
    return chain # Return an invokable chain! Check it out in our langchain_ops router

# Sequential chain that adds an extra step in the to refine the initial response
def get_sequential_chain():

    # First chain - just a basic prompt to the LLM. Using the OG members from above
    draft_chain = prompt | llm

    # Define a new prompt to help us refine the initial answer
    # In this case, we want a more concise and professional answer. No crazy rambling
    refined_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are a stoic and professional chatbot. 
         You take raw LLM answers and refine them to be more concise and professional.
         Format your text generation in 3 or less sentences. 
         Share the refined answer, followed by the original answer"""),
        ("user", "{input}")
    ])

    # Make the second chain using the refined prompt
    refined_chain = refined_prompt | llm

    # Finally, the sequential part - combine the 2 chains and return the final chain!
    sequential_chain = draft_chain | refined_chain
    return sequential_chain

# A Chain that stores memory so it can recall what was being talked about
def get_memory_chain():

    # Create a memory object, an instance of ConversationBufferWindowMemory
    # This Memory instance remembers the last "k" interactions
    memory = ConversationBufferWindowMemory(k=3)

    # Make a Prompt and Chain the old fashioned way... look how clunky

    # Prompt

    # Chain

    # Return the chain with memory, invoked in the router endpoint