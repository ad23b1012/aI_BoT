import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    res = embeddings.embed_query("hello")
    print("Success with models/embedding-001")
except Exception as e:
    print("models/embedding-001 FAILED:", e)

try:
    embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001")
    res = embeddings.embed_query("hello")
    print("Success with embedding-001")
except Exception as e:
    print("embedding-001 FAILED:", e)

try:
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    res = embeddings.embed_query("hello")
    print("Success with models/text-embedding-004")
except Exception as e:
    print("models/text-embedding-004 FAILED:", e)

try:
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    res = embeddings.embed_query("hello")
    print("Success with text-embedding-004")
except Exception as e:
    print("text-embedding-004 FAILED:", e)
