import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    res = embeddings.embed_query("hello")
    print("Success with models/gemini-embedding-001")
except Exception as e:
    print("models/gemini-embedding-001 FAILED:", e)
