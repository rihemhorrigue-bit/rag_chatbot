import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("Error: GROQ_API_KEY not found in .env")
    exit(1)

client = Groq(api_key=api_key)
MODEL = "qwen/qwen3.8-27b"

print(f"Using model: {MODEL}")
print("Type your messages below (type 'exit' or 'quit' to stop):")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": user_input}],
            model=MODEL,
        )
        response = chat_completion.choices[0].message.content
        usage = chat_completion.usage
        
        print(f"AI: {response}")
        print(f"[Tokens used: Prompt={usage.prompt_tokens}, Completion={usage.completion_tokens}, Total={usage.total_tokens}]")
    except Exception as e:
        print(f"Error: {e}")