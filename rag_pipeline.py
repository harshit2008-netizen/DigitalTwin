import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from data_loader import load_documents, get_chunks

CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

def get_embedding_function():
    # Use the local HuggingFace model specified by the user
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def create_vector_db():
    print("Loading and chunking documents...")
    docs = load_documents("APJ_Resources")
    if not docs:
        print("No documents found.")
        return None
    
    chunks = get_chunks(docs)
    print(f"Creating ChromaDB at {CHROMA_PATH}...")
    
    embedding_function = get_embedding_function()
    db = Chroma.from_documents(chunks, embedding_function, persist_directory=CHROMA_PATH)
    db.persist()
    print("Database created successfully.")
    return db

def get_vector_db():
    if os.path.exists(CHROMA_PATH):
        embedding_function = get_embedding_function()
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    else:
        return create_vector_db()

def get_relevant_context(query, k=3):
    db = get_vector_db()
    if not db: return []
    return db.similarity_search(query, k=k)

if __name__ == "__main__":
    print("Testing RAG Pipeline with BAAI/bge-small-en-v1.5...")
    query = "What is the SLV-3?"
    results = get_relevant_context(query, k=2)
    for doc in results:
        print(doc.page_content[:100])