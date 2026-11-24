import os
from pathlib import Path

from dotenv import load_dotenv

from llm_client import GeminiProvider, LLMClient, OllamaProvider


def main():
    env_path = Path(__file__).with_name(".env")
    load_dotenv(dotenv_path=env_path, override=True)

    provider_name = input("Choose provider (gemini/ollama): ").strip().lower()

    if provider_name == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
        if not api_key or api_key == "Give your API Key here…":
            raise ValueError("Gemini API key not found in .env")
        provider = GeminiProvider(api_key)
    elif provider_name == "ollama":
        provider = OllamaProvider(model="llama3.2")
    else:
        raise ValueError("Invalid provider selected. Choose 'gemini' or 'ollama'.")

    client = LLMClient(provider)
    prompt = input("Enter your question: ").strip()
    if not prompt:
        raise ValueError("Please enter a question.")

    answer = client.generate(prompt)
    print("\nLLM Response:")
    print(answer)


if __name__ == "__main__":
    main()
