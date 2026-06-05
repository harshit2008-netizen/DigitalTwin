# Dr. A.P.J. Abdul Kalam - Digital Twin 

This repository contains the codebase for the "Digital Twin of a Scientist" project, serving as an interactive, AI-driven replica of Dr. A.P.J. Abdul Kalam. The application uses a Retrieval-Augmented Generation (RAG) pipeline to maintain strict factual accuracy based on his speeches, technical papers, and biography, while enforcing a visionary and educational persona.

## Architecture and Workflow

The project consists of three main components:

### 1. Data Ingestion (`data_loader.py`)
This module scans the `APJ_Resources/` directory for `.txt` and `.pdf` files. It extracts the raw text and splits it into manageable, overlapping chunks (1000 characters) using LangChain's `RecursiveCharacterTextSplitter`. This chunking strategy ensures that the semantic meaning of Dr. Kalam's aerospace and defense writings is preserved.

### 2. RAG Pipeline & Vector Database (`rag_pipeline.py`)
This module converts the text chunks into mathematical vectors (embeddings) using Google's Generative AI models. These embeddings are stored locally in a `ChromaDB` instance (`/chroma_db/`).
When a user asks a question, this pipeline performs a similarity search to fetch the top 3 most relevant textual chunks to provide factual grounding. 
*Note: This script uses a round-robin API key approach to bypass free-tier rate limits.*

### 3. Conversational Memory & UI (`chatbot_engine.py` & `app.py`)
- **`chatbot_engine.py`**: Houses the core logic for the LangChain interaction. It injects a strict system prompt (defining Dr. Kalam's persona and biographical details) and builds a Conversational RAG Chain. A history-aware retriever reformulates follow-up questions to ensure the conversation flows naturally.
- **`app.py`**: The Streamlit user interface. It manages the chat history state (`st.session_state`) and provides a clean, beautiful web interface to interact with the Digital Twin.

##  How to Run Locally

1. **Prerequisites**: Ensure you have Python 3.10+ installed.
2. **Environment Variables**: Create a `.env` file in the root directory and add your Gemini API keys:
   ```env
   GEMINI_API_KEY_1=your_key_here
   GEMINI_API_KEY_2=your_key_here
   # up to GEMINI_API_KEY_5 for round-robin load balancing
   ```
3. **Install Dependencies**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```
4. **Launch the App**:
   ```powershell
   streamlit run app.py
   ```

Enjoy conversing with the ignited mind of Dr. Kalam!
