# 🤖 AI ChatBot: Startup Legal & Compliance Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](https://aichatbotneo.streamlit.app/)

## 🚀 Live Demo
**Access the live application here:** [https://aichatbotneo.streamlit.app/](https://aichatbotneo.streamlit.app/)

---

## 📌 Project Overview
The **Startup Legal & Compliance Assistant** is an intelligent AI-powered platform designed to help founders navigate the complex landscape of startup law, incorporation, and compliance. By combining curated local knowledge with real-time web search, it provides accurate, context-aware guidelines for early-stage entrepreneurs.

This project was built as part of the NeoStats AI Engineer Challenge to showcase advanced RAG implementation, agentic routing, and premium UX design.

---

## ✨ Key Features

### 🔹 Core Capabilities
- **Hybrid Intelligence:** Seamlessly switches between local document knowledge (RAG) and live web search.
- **Multi-Model Support:** Optimized for **Groq (LLaMA 3.3 70B)** for speed and **Google Gemini 1.5 Flash** for deep reasoning.
- **RAG Implementation:** Context-aware responses based on local legal guides and uploaded documents using FAISS vector store.
- **Real-time Web Search:** Integrated with Tavily Search API to provide up-to-date regulatory information.

### 🌟 "Shine" Features (Advanced)
- **Agentic Routing:** The AI autonomously determines the best tool (Search or RAG) to use based on user intent.
- **Token Streaming:** Real-time response generation for a premium, conversational experience.
- **Dynamic File Uploads:** Drop PDF or TXT files directly into the sidebar to analyze them instantly.
- **Transparent Citations:** View exactly where information came from with interactive source expanders.
- **Integrated TTS:** Listen to assistant responses with high-quality Text-to-Speech.

---

## 🛠️ Tech Stack
- **Frontend/UI:** [Streamlit](https://streamlit.io/)
- **LLM Orchestration:** [LangChain](https://www.langchain.com/)
- **LLM Providers:** [Groq](https://groq.com/), [Google Generative AI](https://ai.google.dev/)
- **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss)
- **Search API:** [Tavily](https://tavily.com/)
- **Embeddings:** Google Text-Embedding-004

---

## ⚙️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd AI_UseCase
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GROQ_API_KEY=your_groq_key
   GOOGLE_API_KEY=your_google_key
   TAVILY_API_KEY=your_tavily_key
   ```

5. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

---

## 📂 Project Structure
- `app.py`: Main Streamlit application and UI logic.
- `models/`: LLM and embedding model initializations.
- `utils/`: Core logic for RAG functionality and Web Search.
- `config/`: Configuration and API key management.
- `data/`: Local knowledge base storage.

---

## 📝 Disclaimer
*This assistant provides standard legal guidelines and compliance information for educational purposes only. It does not constitute formal legal advice.*

---
**Build with ❤️ for the NeoStats AI Engineer Challenge.**
