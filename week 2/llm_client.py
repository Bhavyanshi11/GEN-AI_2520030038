from abc import ABC, abstractmethod

import requests
from google import genai


class LLMProvider(ABC):
    """Common interface implemented by hosted and local LLM providers."""

    @abstractmethod
    def generate(self, prompt):
        """Return a text response for the supplied prompt."""
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    def __init__(self, api_key, model="gemini-3.8-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt):
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return response.text or ""


class OllamaProvider(LLMProvider):
    def __init__(self, model="llama3.2", url="http://localhost:11434/api/generate"):
        self.model = model
        self.url = url

    def generate(self, prompt):
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()


class LLMClient:
    """Provider-independent interface for generating text."""

    def __init__(self, provider):
        if not callable(getattr(provider, "generate", None)):
            raise TypeError("provider must implement a generate(prompt) method")
        self.provider = provider

    def generate(self, prompt):
        return self.provider.generate(prompt)
