import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file explicitly
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=env_path)

def get_api_key(provider):
    """
    Safely fetch an API key for the given provider from Streamlit Secrets or Environment Variables.
    Provider names generally look like 'OPENAI_API_KEY', 'GROQ_API_KEY', etc.
    """
    try:
        # First check streamlit secrets (ideal for cloud deployment)
        if provider in st.secrets:
            return st.secrets[provider]
    except Exception:
        # If secrets file is not available, we catch the exception and fall back
        pass

    # Fallback to local environment variables (which are now populated by .env)
    return os.environ.get(provider, "")

