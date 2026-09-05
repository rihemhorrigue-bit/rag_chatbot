
import os
import sys
from dotenv import load_dotenv
from groq import Groq  # pip install groq

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    print("ERROR: GROQ_API_KEY not found in .env file.")
    sys.exit(1)

# Initialize the Groq client
client = Groq(api_key=API_KEY)

def test_model(model_id: str) -> dict:
    """
    Send a test prompt to a specific model and return response details.
    Returns a dict with:
        - 'success': bool
        - 'response': str (if success)
        - 'prompt_tokens', 'completion_tokens', 'total_tokens' (if success)
        - 'error': str (if failure)
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "hi",
                }
            ],
            model=model_id,
        )
        # Extract usage
        usage = chat_completion.usage
        return {
            "success": True,
            "response": chat_completion.choices[0].message.content,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    print("Groq API Connection Test")
    print("========================")
    print(f"Using API key: {API_KEY[:4]}...{API_KEY[-4:]}\n")

    # Fetch available models
    try:
        models = client.models.list()
        model_ids = [model.id for model in models.data]
    except Exception as e:
        print(f"Failed to fetch model list: {e}")
        sys.exit(1)

    if not model_ids:
        print("No models returned from the API.")
        sys.exit(1)

    print(f"Found {len(model_ids)} models:\n")
    for model_id in model_ids:
        print(f"Testing model: {model_id}")
        result = test_model(model_id)
        if result["success"]:
            print(f"  Response: {result['response']}")
            print(f"  Tokens used: prompt={result['prompt_tokens']}, "
                  f"completion={result['completion_tokens']}, "
                  f"total={result['total_tokens']}")
        else:
            print(f"  ERROR: {result['error']}")
        print()  # blank line between models

if __name__ == "__main__":
    main()