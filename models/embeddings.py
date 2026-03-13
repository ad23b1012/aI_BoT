import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config.config import get_api_key

def get_embedding_model():
    """
    Initialize and return an embedding model for RAG.
    We use Google Gemini's embedding model since it is free.
    """
    try:
        google_key = get_api_key("GOOGLE_API_KEY")
        if google_key:
            return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=google_key)
            
        # If no keys, return None so the app can handle it gracefully.
        return None
        
    except Exception as e:
        print(f"Warning: Failed to initialize embedding model: {str(e)}")
        return None

