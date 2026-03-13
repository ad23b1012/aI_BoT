import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from models.embeddings import get_embedding_model

def get_vector_store(data_folder="data"):
    """
    Loads all text and PDF documents from the specified folder,
    chunks them, and creates a FAISS vector store.
    """
    if not os.path.exists(data_folder):
        return None

    documents = []
    
    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)
        if filename.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
        elif filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

    if not documents:
        return None

    embeddings = get_embedding_model()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Create the FAISS vector store
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        return None

def retrieve_context(vector_store, query, k=3):
    """
    Retrieves the top k most relevant text chunks for the given query.
    """
    if not vector_store:
        return ""
    
    docs = vector_store.similarity_search(query, k=k)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
    return context_text
