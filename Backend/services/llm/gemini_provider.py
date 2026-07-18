from __future__ import annotations

import os
from typing import Iterator, List, Optional, Sequence, Union

from services.llm.base_llm import BaseLLMProvider, ChatMessage, LLMResponse


class GeminiProvider(BaseLLMProvider):
    """
    Optional provider kept for fallback.
    Disabled by default. Import is lazy so the backend can run offline without google-generativeai installed.
    """

    def __init__(self, *, model: str = "gemini-2.5-flash"):
        self.model_name = model
        self.api_key = os.getenv("GEMINI_API_KEY", "")

    def _model(self):
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as e:
            raise RuntimeError("google-generativeai not installed") from e
        genai.configure(api_key=self.api_key)
        return genai.GenerativeModel(self.model_name)

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        top_k: int = 40,
        seed: Optional[int] = 42,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ):
        # No streaming implemented here (kept minimal)
        model = self._model()
        genai = __import__("google.generativeai").generativeai  # type: ignore
        generation_config = genai.types.GenerationConfig(
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            max_output_tokens=int(max_tokens),
        )
        resp = model.generate_content(prompt, generation_config=generation_config)
        return LLMResponse(text=(resp.text or "").strip(), raw={"provider": "gemini"})

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        top_k: int = 40,
        seed: Optional[int] = 42,
        stream: bool = False,
    ):
        # Simple formatting for Gemini as single prompt
        prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
        return self.generate(prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p, top_k=top_k)

    def summarize(
        self,
        text: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 512,
        stream: bool = False,
    ):
        prompt = f"Summarize:\n\n{text}"
        return self.generate(prompt, temperature=temperature, max_tokens=max_tokens)

