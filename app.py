import streamlit as st
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# Add local directories to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.llm import get_llm_model
from utils.rag import get_vector_store, retrieve_context
from utils.search import perform_web_search


def get_chat_response(chat_model, messages, system_prompt, use_rag=False, rag_vs=None, use_search=False, mode="Detailed"):
    """Get response from the chat model"""
    try:
        latest_message = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        additional_context = ""

        if use_rag and rag_vs and latest_message:
            docs_context = retrieve_context(rag_vs, latest_message)
            if docs_context:
                additional_context += f"\n\n--- DOCUMENT CONTEXT ---\n{docs_context}\n"

        if use_search and latest_message:
            search_context = perform_web_search(latest_message)
            if search_context:
                additional_context += f"\n\n--- WEB SEARCH RESULTS ---\n{search_context}\n"

        if additional_context:
            system_prompt += f"\n\nPlease use the following context to answer the user if relevant. If it's not relevant, ignore it.{additional_context}"

        if mode == "Concise":
            system_prompt += "\n\nINSTRUCTION: Provide a very brief, concise, and summarized response."
        else:
            system_prompt += "\n\nINSTRUCTION: Provide a highly detailed, comprehensive, and in-depth response."

        # Prepare messages for the model
        formatted_messages = [SystemMessage(content=system_prompt)]
        
        # Add conversation history
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))
        
        # Get response from model
        response = chat_model.invoke(formatted_messages)
        return response.content
    
    except Exception as e:
        return f"Error getting response: {str(e)}"

def instructions_page():
    """Instructions and setup page"""
    st.title("The Chatbot Blueprint")
    st.markdown("Welcome! Follow these instructions to set up and use the chatbot.")
    
    st.markdown("""
    ## 🔧 Installation
                
    
    First, install the required dependencies: (Add Additional Libraries base don your needs)
    
    ```bash
    pip install -r requirements.txt
    ```
    
    ## API Key Setup
    
    You'll need API keys from your chosen provider. Get them from:
    
    ### OpenAI
    - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
    - Create a new API key
    - Set the variables in config
    
    ### Groq
    - Visit [Groq Console](https://console.groq.com/keys)
    - Create a new API key
    - Set the variables in config
    
    ### Google Gemini
    - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
    - Create a new API key
    - Set the variables in config
    
    ## 📝 Available Models
    
    ### OpenAI Models
    Check [OpenAI Models Documentation](https://platform.openai.com/docs/models) for the latest available models.
    Popular models include:
    - `gpt-4o` - Latest GPT-4 Omni model
    - `gpt-4o-mini` - Faster, cost-effective version
    - `gpt-3.5-turbo` - Fast and affordable
    
    ### Groq Models
    Check [Groq Models Documentation](https://console.groq.com/docs/models) for available models.
    Popular models include:
    - `llama-3.3-70b-versatile` - Large, powerful model
    - `llama-3.1-8b-instant` - Fast, smaller model
    - `mixtral-8x7b-32768` - Good balance of speed and capability
    
    ### Google Gemini Models
    Check [Gemini Models Documentation](https://ai.google.dev/gemini-api/docs/models/gemini) for available models.
    Popular models include:
    - `gemini-1.5-pro` - Most capable model
    - `gemini-1.5-flash` - Fast and efficient
    - `gemini-pro` - Standard model
    
    ## How to Use
    
    1. **Go to the Chat page** (use the navigation in the sidebar)
    2. **Start chatting** once everything is configured!
    
    ## Tips
    
    - **System Prompts**: Customize the AI's personality and behavior
    - **Model Selection**: Different models have different capabilities and costs
    - **API Keys**: Can be entered in the app or set as environment variables
    - **Chat History**: Persists during your session but resets when you refresh
    
    ## Troubleshooting
    
    - **API Key Issues**: Make sure your API key is valid and has sufficient credits
    - **Model Not Found**: Check the provider's documentation for correct model names
    - **Connection Errors**: Verify your internet connection and API service status
    
    ---
    
    Ready to start chatting? Navigate to the **Chat** page using the sidebar! 
    """)

def chat_page():
    """Main chat interface page"""
    st.title("🤖 AI ChatBot")
    
    # Get configuration from environment variables or session state
    # Default system prompt
    system_prompt = "You are a Startup Legal & Compliance Assistant. Help founders navigate incorporation, IP, employment law, and general legal operations. Do not provide formal legal advice but offer standard knowledge guidelines."
    
    # Initialize RAG in session state
    if "rag_vs" not in st.session_state:
        with st.spinner("Initializing Knowledge Base..."):
            st.session_state.rag_vs = get_vector_store()
    
    with st.sidebar:
        st.divider()
        st.subheader("Assistant Settings")
        use_rag = st.checkbox("📚 Enable Local Knowledge (RAG)", value=True)
        use_search = st.checkbox("🌐 Enable Live Web Search", value=False)
        response_mode = st.radio("Response Mode", ["Concise", "Detailed"], index=1)

    # Determine which provider to use based on available API keys
    chat_model = get_llm_model()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if not chat_model:
        st.warning("⚠️ No valid LLM API keys found. Please add your `GROQ_API_KEY` (and `TAVILY_API_KEY` / `GOOGLE_API_KEY`) to proceed. Check the Instructions tab.")
        return

    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display bot response
        with st.chat_message("assistant"):
            with st.spinner("Getting response..."):
                response = get_chat_response(
                    chat_model, 
                    st.session_state.messages, 
                    system_prompt,
                    use_rag=use_rag,
                    rag_vs=st.session_state.rag_vs,
                    use_search=use_search,
                    mode=response_mode
                )
                st.markdown(response)
        
        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    st.set_page_config(
        page_title="LangChain Multi-Provider ChatBot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Go to:",
            ["Chat", "Instructions"],
            index=0
        )
        
        # Add clear chat button in sidebar for chat page
        if page == "Chat":
            st.divider()
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    
    # Route to appropriate page
    if page == "Instructions":
        instructions_page()
    if page == "Chat":
        chat_page()

if __name__ == "__main__":
    main()