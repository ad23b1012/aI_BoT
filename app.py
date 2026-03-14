import streamlit as st
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# Add local directories to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.llm import get_llm_model
from utils.rag import get_vector_store, retrieve_context
from utils.search import perform_web_search
from utils.agent import get_agent_executor
import asyncio
import edge_tts
import tempfile


async def generate_speech(text):
    """Generate speech using edge-tts"""
    voice = "en-US-GuyNeural"
    communicate = edge_tts.Communicate(text, voice)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        await communicate.save(tmp_file.name)
        return tmp_file.name

def get_chat_response(chat_model, messages, system_prompt):
    """Get response from the Agent Executor"""
    try:
        agent_executor = get_agent_executor(chat_model, system_prompt)
        
        # Prepare history
        history = []
        for msg in messages[:-1]: # All but the latest user msg
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            else:
                history.append(AIMessage(content=msg["content"]))
        
        latest_input = messages[-1]["content"] if messages else ""
        
        # Using invoke since streaming an agent requires more complex event handling
        # But we'll use st.spinner to keep UX smooth
        response = agent_executor.invoke({
            "input": latest_input,
            "chat_history": history
        })
        
        return response["output"]
    
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
        
        st.divider()
        st.subheader("📁 Knowledge Base")
        uploaded_files = st.file_uploader("Upload legal docs (PDF/TXT)", type=['pdf', 'txt'], accept_multiple_files=True)
        
        if uploaded_files:
            if st.button("🚀 Process & Index Uploads", use_container_width=True):
                with st.spinner("Indexing documents..."):
                    # Save files to a temp directory and index them
                    upload_dir = "data" # Use existing data folder
                    if not os.path.exists(upload_dir):
                        os.makedirs(upload_dir)
                    
                    for uploaded_file in uploaded_files:
                        with open(os.path.join(upload_dir, uploaded_file.name), "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    # Force recreate vector store to include new uploads
                    st.session_state.rag_vs = get_vector_store(data_folder=upload_dir)
                    st.success("Knowledge base updated!")
                    st.rerun()

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
            with st.spinner("Assistant is thinking and choosing tools..."):
                # Clear previous sources
                st.session_state.sources_used = []
                
                response = get_chat_response(
                    chat_model, 
                    st.session_state.messages, 
                    system_prompt
                )
            
            # Typewriter effect for agent response to maintain "streaming" feel
            response_placeholder = st.empty()
            full_response = ""
            import time
            for word in response.split(" "):
                full_response += word + " "
                response_placeholder.markdown(full_response + "▌")
                time.sleep(0.05)
            response_placeholder.markdown(full_response)
                
            # Display citations if sources were tracked during agent run
            if st.session_state.get("sources_used"):
                with st.expander("🔍 View Sources"):
                    for source in set(st.session_state.sources_used):
                        st.info(source)
                
            # TTS Button
            if st.button("🔊 Listen to response"):
                with st.spinner("Generating audio..."):
                    audio_file = asyncio.run(generate_speech(response))
                    st.audio(audio_file, format="audio/mp3", autoplay=True)
        
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