from langchain_classic.agents import AgentExecutor, initialize_agent, AgentType
from langchain_core.tools import tool
from utils.rag import retrieve_context
from utils.search import perform_web_search
import streamlit as st

@tool
def search_web(query: str):
    """Useful for finding current information about the world, news, or specific startup regulations. Use this if the knowledge base doesn't have the answer."""
    if "sources_used" not in st.session_state:
        st.session_state.sources_used = []
    st.session_state.sources_used.append("🌐 Real-time Web Search")
    return perform_web_search(query)

@tool
def lookup_knowledge_base(query: str):
    """Useful for finding internal startup legal guidelines, incorporation steps, and compliance docs from the local database."""
    if "sources_used" not in st.session_state:
        st.session_state.sources_used = []
    st.session_state.sources_used.append("📄 Internal Knowledge Base")
    # We get the vector store from session state since it might be dynamic
    rag_vs = st.session_state.get("rag_vs")
    if not rag_vs:
        return "Knowledge base not initialized."
    return retrieve_context(rag_vs, query)

def get_agent_executor(llm, system_prompt):
    """Initialize and return a LangChain Agent Executor using the classic initialize_agent"""
    tools = [search_web, lookup_knowledge_base]
    
    # Using CHAT_CONVERSATIONAL_REACT_DESCRIPTION which is very robust for various LLMs
    agent_executor = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="generate",
        system_message=system_prompt
    )
    
    return agent_executor

