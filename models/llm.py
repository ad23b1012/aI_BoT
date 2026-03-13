import os
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import get_api_key

def get_llm_model():
    """Initialize and return an LLM (Groq or Gemini)"""
    try:
        groq_key = get_api_key("GROQ_API_KEY")
        if groq_key:
            return ChatGroq(
                api_key=groq_key,
                model="llama-3.3-70b-versatile",
            )
            
        google_key = get_api_key("GOOGLE_API_KEY")
        if google_key:
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=google_key
            )
            
        return None
    except Exception as e:
        print(f"Warning: Failed to initialize LLM model: {str(e)}")
        return None