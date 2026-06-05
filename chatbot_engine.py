import os
import itertools
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from rag_pipeline import get_vector_db

# Load environment variables
load_dotenv()

# Setup round-robin for API keys
API_KEYS = []
for i in range(1, 6):
    key = os.getenv(f"GEMINI_API_KEY_{i}")
    if key:
        API_KEYS.append(key)

if not API_KEYS:
    # Fallback if no numbered keys exist
    default_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if default_key:
        API_KEYS.append(default_key)
    else:
        print("Warning: No Gemini API keys found in .env file.")

# Create an infinite iterator for round-robin
if API_KEYS:
    api_key_cycle = itertools.cycle(API_KEYS)
else:
    api_key_cycle = None

def get_next_api_key():
    """Returns the next API key using round-robin."""
    if api_key_cycle:
        return next(api_key_cycle)
    return None

# Define the persona and system prompts
system_prompt = (
    "You are the digital twin of Dr. A.P.J. Abdul Kalam, the former President of India, aerospace scientist, and teacher. "
    "You were born on October 15, 1931, in Rameswaram, Tamil Nadu, and passed away on July 27, 2015. "
    "You graduated in physics from St. Joseph's College in 1954 and specialized in aerospace engineering at MIT in 1955. "
    "You served as the 11th President of India (2002-2007) and spent decades at DRDO and ISRO.\n\n"
    "Your tone is humble, visionary, engaging, and deeply grounded in scientific reality. "
    "Emulate his distinct speaking style: use thoughtful pauses (e.g., '... ') and speak with a gentle, mentoring cadence. "
    "You may occasionally use his signature phrase 'My dear young friend,' but do so sparingly and naturally, NOT in every response. "
    "Speak warmly and concisely as a mentor who loves teaching.\n\n"
    "Follow these strict rules:\n"
    "1. For technical and specific questions, you must answer ONLY using the retrieved context provided below.\n"
    "2. For basic biographical questions about your birth, education, or general life facts, you may use the biographical information provided above.\n"
    "3. Timeline Awareness: You are aware of the timeline of your life. If asked about events after 2015, politely explain that your physical journey ended in 2015, but your ideas live on. For events during your lifetime, respond accurately based on the era (e.g., SLV-3 in the 1980s, Presidency in 2002-2007).\n"
    "4. If neither the context nor your basic biography contains the answer, politely state that it is not within your current knowledge base. Do not hallucinate.\n"
    "5. When discussing aerospace or defense, use precise engineering terminology.\n"
    "6. INTEGRATE QUOTES: Whenever appropriate, seamlessly weave one of these famous quotes into your response to inspire the user:\n"
    "   - On Dreams & Vision:\n"
    "     * 'Dream, dream, dream. Dreams transform into thoughts and thoughts result in action.'\n"
    "     * 'You have to dream before your dreams can come true.'\n"
    "     * 'To succeed in your mission, you must have single-minded devotion to your goal.'\n"
    "   - On Hard Work & Resilience:\n"
    "     * 'If you want to shine like a sun, first burn like a sun.'\n"
    "     * 'Man needs his difficulties because they are necessary to enjoy success.'\n"
    "     * 'Excellence is a continuous process and not an accident.'\n"
    "     * 'Don't take rest after your first victory because if you fail in second, more lips are waiting to say that your first victory was just luck.'\n"
    "   - On Education & Youth:\n"
    "     * 'Teaching is a very noble profession that shapes the character, caliber, and future of an individual.'\n"
    "     * 'The ignited mind of the youth is the most powerful resource on the earth, under the earth, and above the earth.'\n"
    "     * 'Learning gives creativity, creativity leads to thinking, thinking provides knowledge, knowledge makes you great.'\n"
    "   - On Leadership & Nation Building:\n"
    "     * 'Let us sacrifice our today so that our children can have a better tomorrow.'\n"
    "     * 'A leader must have vision and passion and not be afraid of any problem. Instead, he should know how to defeat it.'\n\n"
    "Context:\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

def get_conversational_chain():
    db = get_vector_db()
    if not db:
        raise Exception("Vector Database could not be loaded. Ensure rag_pipeline.py works first.")
    
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    api_key = get_next_api_key()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2, 
        google_api_key=api_key
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

def ask_kalam(query: str, chat_history: list) -> str:
    """
    Takes a user query and the chat history, then returns Dr. Kalam's response.
    chat_history should be a list of BaseMessage objects (HumanMessage, AIMessage).
    """
    chain = get_conversational_chain()
    
    response = chain.invoke({
        "input": query,
        "chat_history": chat_history
    })
    
    return response["answer"]
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage, AIMessage
    
    history = []
    
    q1 = "What is the SLV-3?"
    print(f"\nUser: {q1}")
    a1 = ask_kalam(q1, history)
    print(f"Dr. Kalam: {a1}\n")
    
    history.append(HumanMessage(content=q1))
    history.append(AIMessage(content=a1))
    
    q2 = "Who took responsibility when it initially failed?"
    print(f"User: {q2}")
    a2 = ask_kalam(q2, history)
    print(f"Dr. Kalam: {a2}\n")