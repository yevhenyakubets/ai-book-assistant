import logging
import os

import openai
from openai import OpenAI

from ai_book_assistant.llm.errors import LLMError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5.4-mini"
# The most one call may generate. This bounds the cost of a runaway answer;
# reasoning models count their hidden reasoning tokens against it.
MAX_OUTPUT_TOKENS = 4000
# OpenAI reports an empty prepaid balance as a 429 with this error code.
INSUFFICIENT_QUOTA = "insufficient_quota"
PAYMENT_REQUIRED = 402
TEMPERATURE = 0.0


def _status_of(exc: openai.APIStatusError) -> int:
    """The HTTP status, except that an empty balance is not a rate limit."""
    if exc.status_code == 429 and getattr(exc, "code", None) == INSUFFICIENT_QUOTA:
        return PAYMENT_REQUIRED
    return exc.status_code


class OpenAILLMClient:
    def __init__(
        self,
        model: str | None = None,
        *,
        client: object | None = None,
        max_retries: int = 3,
        timeout: float = 30.0,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        temperature: float | None = TEMPERATURE,
    ) -> None:
        if client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is missing.")
            # The SDK retries connection errors, 429 and 5xx with backoff.
            client = OpenAI(api_key=api_key, max_retries=max_retries, timeout=timeout)
        self.client = client
        self.model = model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    def generate(self, prompt: str) -> str:
        try:
            response = self._create(prompt)
        except openai.BadRequestError as exc:
            if self.temperature is None or getattr(exc, "param", None) != "temperature":
                raise self._as_llm_error(exc) from exc
            logger.warning(
                "%s does not accept a temperature; retrying without one", self.model
            )
            self.temperature = None
            return self.generate(prompt)
        except openai.APIStatusError as exc:
            raise self._as_llm_error(exc) from exc
        except openai.APIConnectionError as exc:  # includes timeouts
            logger.error("Could not reach OpenAI: %s", exc.message)
            raise LLMError(f"Could not reach OpenAI: {exc.message}") from exc
        self._log_usage(response)
        return response.choices[0].message.content

    def _create(self, prompt: str):
        options = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        return self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_completion_tokens=self.max_output_tokens,
            **options,
        )

    @staticmethod
    def _as_llm_error(exc: openai.APIStatusError) -> LLMError:
        status = _status_of(exc)
        logger.error("OpenAI returned HTTP %s: %s", status, exc.message)
        return LLMError(
            f"OpenAI returned HTTP {status}: {exc.message}", status_code=status
        )

    def _log_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage is not None:
            details = getattr(usage, "completion_tokens_details", None)
            logger.info(
                "OpenAI usage (%s): %s in, %s out, %s reasoning",
                self.model,
                usage.prompt_tokens,
                usage.completion_tokens,
                getattr(details, "reasoning_tokens", None),
            )
        if getattr(response.choices[0], "finish_reason", None) == "length":
            logger.warning(
                "OpenAI stopped at the %d-token output limit", self.max_output_tokens
            )
