from langchain_tavily import TavilySearch
from langchain_community.tools import DuckDuckGoSearchRun
from config.config import get_api_key

def perform_web_search(query: str) -> str:
    """
    Perform a live web search using Tavily. 
    Falls back to DuckDuckGo if Tavily API key is missing.
    """
    tavily_api_key = get_api_key("TAVILY_API_KEY")
    
    if tavily_api_key:
        try:
            # We explicitly pass the key or ensure it exists in environment
            search_tool = TavilySearch(max_results=3, tavily_api_key=tavily_api_key)
            results = search_tool.invoke({"query": query})
            # Tavily returns a list of dicts: [{'url': '...', 'content': '...'}, ...]
            if isinstance(results, list):
                context = ""
                for r in results:
                    context += f"Source: {r.get('url', 'Unknown')}\nSnippet: {r.get('content', '')}\n\n"
                return context
            return str(results)
        except Exception as e:
            print(f"Tavily search failed: {e}. Falling back to DuckDuckGo...")
            
    # Fallback
    try:
        ddg = DuckDuckGoSearchRun()
        return ddg.invoke(query)
    except Exception as e:
        return f"Web search failed: {str(e)}"
