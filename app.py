import streamlit as st
import json
import os
from gtts import gTTS
import base64
from langchain_core.messages import HumanMessage, AIMessage
from chatbot_engine import ask_kalam

MEMORY_FILE = "long_term_memory.json"

st.set_page_config(
    page_title="Dr. A.P.J. Abdul Kalam - Digital Twin",
    page_icon="🚀",
    layout="wide"
)

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_memory(messages):
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(messages, f, indent=4)

def generate_audio(text):
    tts = gTTS(text=text, lang='en', tld='co.in') # Indian English accent
    audio_file = "response.mp3"
    tts.save(audio_file)
    return audio_file

# Sidebar setup: Memory Visualization and Project Details
with st.sidebar:
    st.image("kalam.jpg", width=150)
    st.title("About the Project")
    st.info("AIMS DTU Summer Project 2026: Digital Twin of Dr. A.P.J. Abdul Kalam.")
    
    st.subheader("🧠 Memory Visualization (Bonus)")
    if st.button("Clear Long-Term Memory"):
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)
        st.session_state.messages = []
        st.rerun()
    
    with st.expander("View Raw Memory Data"):
        memory_data = load_memory()
        st.json(memory_data)

st.title("🚀 Dr. A.P.J. Abdul Kalam - Digital Twin")
st.markdown("*\"Dream, dream, dream. Dreams transform into thoughts and thoughts result in action.\"*")

# Initialize chat history from Long-Term Memory
if "messages" not in st.session_state or not st.session_state.messages:
    loaded_mem = load_memory()
    if loaded_mem:
        st.session_state.messages = loaded_mem
    else:
        st.session_state.messages = [
            {"role": "assistant", "content": "Greetings! I am the digital twin of Dr. A.P.J. Abdul Kalam. How may I guide your ignited mind today?"}
        ]

# Define avatar dictionary
AVATARS = {
    "user": "🧑‍🎓",
    "assistant": "kalam.jpg"
}

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATARS.get(message["role"])):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask Dr. Kalam a question..."):
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(prompt)
        
    # Convert session state messages to LangChain message objects
    langchain_history = []
    for msg in st.session_state.messages[1:]:
        if msg["role"] == "user":
            langchain_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_history.append(AIMessage(content=msg["content"]))

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display response
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("Reflecting on the knowledge base..."):
            try:
                response = ask_kalam(prompt, langchain_history)
                st.markdown(response)
                
                # Bonus: Voice Interaction
                with st.spinner("Generating Voice..."):
                    audio_path = generate_audio(response)
                    st.audio(audio_path, format="audio/mp3")

                st.session_state.messages.append({"role": "assistant", "content": response})
                # Save to long-term memory
                save_memory(st.session_state.messages)
            except Exception as e:
                st.error(f"An error occurred: {e}")
