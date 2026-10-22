import logging
import os

import groq
from groq import Groq

from ai_book_assistant.llm.errors import LLMError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-120b"


class GroqLLMClient:
    def __init__(
        self,
        model: str | None = None,
        *,
        client: object | None = None,
        max_retries: int = 3,
        timeout: float = 30.0,
    ) -> None:
        if client is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable is missing.")
            # The SDK retries connection errors, 429 and 5xx with backoff.
            client = Groq(api_key=api_key, max_retries=max_retries, timeout=timeout)
        self.client = client
        self.model = model or os.getenv("GROQ_MODEL") or DEFAULT_MODEL

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
        except groq.APIStatusError as exc:
            logger.error("Groq returned HTTP %s: %s", exc.status_code, exc.message)
            raise LLMError(
                f"Groq returned HTTP {exc.status_code}: {exc.message}",
                status_code=exc.status_code,
            ) from exc
        except groq.APIConnectionError as exc:  # includes timeouts
            logger.error("Could not reach Groq: %s", exc.message)
            raise LLMError(f"Could not reach Groq: {exc.message}") from exc
        return response.choices[0].message.content
