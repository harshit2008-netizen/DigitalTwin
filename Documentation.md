# Digital Twin of Dr. A.P.J. Abdul Kalam - Documentation

## Approach and Design Decisions

This project aims to build an AI agent that faithfully emulates Dr. A.P.J. Abdul Kalam, matching his knowledge, reasoning style, and highly visionary yet humble persona. To achieve this, the architecture rests on three pillars:

### 1. Retrieval-Augmented Generation (RAG)
To prevent hallucinations and ensure technical accuracy, the agent relies on a RAG pipeline.
*   **Sources**: His autobiography ("Wings of Fire"), speeches, technical papers from the Indian Academy of Sciences (IAS) and DRDO Defence Science Journal (covering topics like SLV-3, Agni missile, carbon-carbon composites, and fluid mechanics).
*   **Vector Database**: We used `ChromaDB` configured with `GoogleGenerativeAIEmbeddings`. To handle API rate limits natively, a robust retry and round-robin load balancer was implemented across multiple Gemini API keys in the backend (`rag_pipeline.py`).

### 2. Conversational Memory Layer
The agent must support multi-turn conversations and remember previous contexts.
*   **Short-Term Memory**: LangChain's `create_history_aware_retriever` intercepts follow-up questions and rewrites them into standalone queries based on the active chat history.
*   **Long-Term Memory**: The Streamlit frontend (`app.py`) serializes the conversation state to `long_term_memory.json`. This ensures that even if the app restarts, Dr. Kalam remembers the user across sessions.

### 3. Persona Design & Timeline Awareness
*   **LLM Model**: We utilize `gemini-2.5-flash` (`ChatGoogleGenerativeAI`), configured with a low temperature (0.2) to ensure high factual consistency.
*   **Prompt Engineering**: The system prompt injects his fundamental biography (birth in Rameswaram, graduation from MIT, presidency) to handle basic questions instantly. It strictly forbids repetitive greetings (like "My dear young friend") to make interactions feel more natural and fluid.
*   **Timeline Awareness**: The prompt explicitly instructs the twin that its physical journey ended in 2015. It handles questions about modern events gracefully by reflecting on them through the lens of his historical knowledge.

### 4. Bonus Deliverables Achieved
1.  **Voice Interaction**: The frontend uses `gTTS` to generate MP3 audio of Dr. Kalam's responses, utilizing an Indian English locale for an authentic feel.
2.  **Memory Visualization Dashboard**: A sidebar expander was added to the UI, allowing developers to inspect the raw JSON of the agent's long-term memory state.
3.  **Timeline Awareness**: Built directly into the system prompt constraints.
