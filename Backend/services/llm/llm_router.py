from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from services.llm.base_llm import BaseLLMProvider, ChatMessage, LLMResponse
from services.llm.ollama_provider import OllamaProvider


class LLMRouter:
    """
    Router that selects provider based on env/config.

    Env:
      LLM_PROVIDER=ollama|gemini
      OLLAMA_MODEL=llama3.1:8b
      OLLAMA_BASE_URL=http://localhost:11434
      LLM_TIMEOUT_SECONDS=30
      LLM_MAX_RETRIES=2
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider_name = (provider or os.getenv("LLM_PROVIDER", "ollama")).lower().strip()
        timeout_s = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
        retries = int(os.getenv("LLM_MAX_RETRIES", "2"))

        if self.provider_name == "gemini":
            from services.llm.gemini_provider import GeminiProvider
            self.provider: BaseLLMProvider = GeminiProvider()
        else:
            self.provider = OllamaProvider(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
                timeout_s=timeout_s,
                max_retries=retries,
            )

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        out = self.provider.generate(prompt, **kwargs)
        if isinstance(out, LLMResponse):
            return out
        # streaming iterator → join
        text = "".join(list(out))
        return LLMResponse(text=text, raw={"streamed": True})

    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        out = self.provider.generate(prompt, stream=True, **kwargs)
        if isinstance(out, LLMResponse):
            yield out.text
            return
        yield from out

    def chat(self, messages: Sequence[ChatMessage], **kwargs) -> LLMResponse:
        out = self.provider.chat(messages, **kwargs)
        if isinstance(out, LLMResponse):
            return out
        text = "".join(list(out))
        return LLMResponse(text=text, raw={"streamed": True})

    def chat_stream(self, messages: Sequence[ChatMessage], **kwargs) -> Iterator[str]:
        out = self.provider.chat(messages, stream=True, **kwargs)
        if isinstance(out, LLMResponse):
            yield out.text
            return
        yield from out

    def summarize(self, text: str, **kwargs) -> LLMResponse:
        out = self.provider.summarize(text, **kwargs)
        if isinstance(out, LLMResponse):
            return out
        s = "".join(list(out))
        return LLMResponse(text=s, raw={"streamed": True})

    @staticmethod
    def safe_json_extract(text: str) -> Any:
        if not text:
            return {}
        raw = text.strip()
        raw = re.sub(r"```json\s*|\s*```", "", raw)
        try:
            return json.loads(raw)
        except Exception:
            pass
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", raw, re.DOTALL)
        if not m:
            return {}
        try:
            return json.loads(m.group(1))
        except Exception:
            return {}

