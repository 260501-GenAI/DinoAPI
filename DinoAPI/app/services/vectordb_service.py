import hashlib

from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# This service will help us initialize and interact with a ChromaDB vector store
# Remember, ChromaDB is just a type of VectorDB. There are others like pinecone

PERSIST_DIRECTORY = "app/chroma_store" # This is where our vectorDB will live

# The vector embedding model we installed
# DIFFERENT from our LLM! This one specializes in turning text into vectors
EMBEDDING = OllamaEmbeddings(model="nomic-embed-text")

# The actual Vector store (which stores our embeddings and lets us interact with them)
# We will initialize it as a dict which lets us manage multiple stores at once
vector_store: dict[str, Chroma] = {}

# A function that gets an instance of the Chroma Dino Vector Store
    #(We could have other stores but this app will just have one for dino documents)
# Similar to how we needed get_db() in the db_connection service
def get_dino_vector_store() -> Chroma:

    # Get (or create) the vector store from the global dict (vector_store)
    if "dino_docs" not in vector_store:
        vector_store["dino_docs"] = Chroma(
            collection_name = "dino_docs", # remember a collection is just a grouping of embeddings
            embedding_function = EMBEDDING,
            persist_directory = PERSIST_DIRECTORY
        )

    # Return the vector store instance, which either already existed or got created in the if statement
    return vector_store["dino_docs"]


# A function that ingests documents into the vector store
# (this is where text gets turned into vectors and stored in the DB)
def ingest_text(text:str):

    """
    This is gonna be a lot - to ingest text we need to:
        1. Clean up the input (remove whitespace etc.)
        2. "Chunk" the data. Split it into smaller pieces for better embedding
        3. Create metadata for the chunks (IDs, importantly)
        4. Embed the chunks (turn them into vectors)
        5. Store the vectors and metadata in the DB
    """

    # Clean the text - no whitespace
    text = text.strip()

    # Chunk the text using a LangChain Transformer, which returns the chunks as a list of stings \
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500, # Each chunk will contain 500 characters
        chunk_overlap = 100, # Each chunk will overlap with its neighbor by 100 chars. Good for retaining context
    )
    chunks = splitter.split_text(text)

    # Fill this list with the enumerated chunks we'll make below
    documents = []
    # Another list for ID of chunks (which we need for ingestion)
    chunk_ids = []

    # Enumeration gives us an (index, value) pair when iterating over a list
    for index, chunk in enumerate(chunks):

        # Generate and attach an ID to each chunk, then add it to the documents list
        documents.append({
            # ID is chunk_index + a hash of the chunk text (to ensure uniqueness)
            "id": f"chunk_{index}_{hashlib.md5(chunk.encode("utf-8")).hexdigest()[:8]}",
            "text": chunk,
        })

        # Generate and attach the IDs to the chunk_ids list
        chunk_ids.append(f"chunk_{index}_{hashlib.md5(chunk.encode("utf-8")).hexdigest()[:8]}")

    # Get the vector store instance
    store = get_dino_vector_store()

    # Turn the documents into a list of LangChain Document object (vectorDB needs this)
    vector_docs = [
        Document(page_content=doc["text"]) for doc in documents
    ]

    # Ingest the documents! (turning them into vectors and storing them)
    store.add_documents(vector_docs, ids=chunk_ids)

    # Return a message with amount of chunks ingested
    return len(documents)


# A function that performs a similarity search on the vector store
# Take the user input, turn it into a vector, and compare it to the vectors in the DB
def search_dino_store(query:str, k:int=6):

    # Get the vector store instance
    store = get_dino_vector_store()

    # Get and save the results of the similarity search (finding the most relevant docs)
    results = store.similarity_search_with_score(query, k=k)

    # Return the results
    return [
        {
            "text": result[0].page_content, # The chunk text
            "score": result[1] # The similarity score (lower is more similar)
        }
        for result in results
    ]