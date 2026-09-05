import os
import sys
from dotenv import load_dotenv
from groq import Groq
from embedder_saver import EmbedderSaver
# 1. Load Environment Variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("❌ ERROR: GROQ_API_KEY not found in .env file.")
    sys.exit(1)

# 2. Configuration
DB_PATH = "./chroma_db"
COLLECTION_NAME = "small_business_kb"

# ✅ Choose the most token‑efficient model for RAG (no complex reasoning)
PREFERRED_MODEL = "qwen/qwen3.8-27b"   # from your tests: 23 tokens for "hi"
# Fallback if preferred model is not available (e.g., after API updates)
FALLBACK_MODEL = "groq/compound-mini"  # also efficient (480 tokens, but works)

# 3. Initialize Clients
client = Groq(api_key=GROQ_API_KEY)
db_engine = EmbedderSaver(db_path=DB_PATH, collection_name=COLLECTION_NAME)

# 4. Validate and set the actual model to use
def get_available_model():
    try:
        models = client.models.list()
        available_ids = [m.id for m in models.data]
    except Exception as e:
        print(f"⚠️ Could not fetch model list: {e}")
        return PREFERRED_MODEL  # hope for the best

    if PREFERRED_MODEL in available_ids:
        return PREFERRED_MODEL
    elif FALLBACK_MODEL in available_ids:
        print(f"⚠️ Preferred model '{PREFERRED_MODEL}' not found. Using fallback '{FALLBACK_MODEL}'.")
        return FALLBACK_MODEL
    else:
        # If neither is available, pick the first chat model from the list
        for m in available_ids:
            if "chat" in m or "llama" in m or "qwen" in m:
                print(f"⚠️ Using first available chat model: {m}")
                return m
        raise RuntimeError("No suitable chat model found in your Groq account.")

MODEL = get_available_model()
print(f"✅ Using model: {MODEL}")

# 5. System Prompt (unchanged)
SYSTEM_PROMPT = """
You are a highly professional Small Business Consultant specializing in SBA Loans and IRS Tax regulations.
Your goal is to provide accurate, concise, and helpful advice based ONLY on the provided context from official documents.

GUIDELINES:
1. Use the provided context to answer the question. 
2. If the answer is not in the context, say: "I'm sorry, but I don't have enough information in the official documents to answer that."
3. Always cite your sources. Mention the Document Title and Page Number if available in the context.
4. Keep the tone professional but accessible for a small business owner.
5. If there are tables or lists in the context, present them clearly.
"""

def get_context(query: str):
    """Retrieves relevant chunks from ChromaDB and formats them for the LLM."""
    results = db_engine.query(query, n_results=5)
    
    context_parts = []
    if not results['documents'][0]:
        return ""

    for i in range(len(results['documents'][0])):
        text = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        source = meta.get('document_title', 'Unknown Document')
        page = meta.get('page_start', 'N/A')
        
        context_parts.append(f"--- SOURCE: {source} (Page {page}) ---\n{text}")
    
    return "\n\n".join(context_parts)

def chat():
    print(f"🚀 RAG Chatbot Started (Using {MODEL})")
    print("Ask me anything about SBA Loans or Tax Guides. Type 'exit' to quit.\n")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        print("\n🔍 Searching knowledge base...")
        context = get_context(user_input)

        if not context:
            print("AI: I couldn't find any relevant information in the uploaded documents.")
            continue

        # Build the messages for the LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT FROM DOCUMENTS:\n{context}\n\nUSER QUESTION: {user_input}"}
        ]

        try:
            print("🤖 Generating answer...\n")
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=MODEL,
                temperature=0.2,
            )

            response = chat_completion.choices[0].message.content
            print(f"ASSISTANT:\n{response}")
            print("-" * 50 + "\n")

        except Exception as e:
            print(f"⚠️ Error from Groq: {e}\n")

if __name__ == "__main__":
    chat()