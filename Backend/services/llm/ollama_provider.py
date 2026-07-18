from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import httpx

from services.llm.base_llm import BaseLLMProvider, ChatMessage, LLMResponse


class OllamaProvider(BaseLLMProvider):
    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: Optional[str] = None,
        timeout_s: float = 30.0,
        max_retries: int = 2,
    ):
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=httpx.Timeout(self.timeout_s))

    def _request_json(self, method: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._client() as c:
                    resp = c.request(method, url, json=payload)
                # Retry on rate limit / transient errors
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(0.4 * (attempt + 1))
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(0.4 * (attempt + 1))
                    continue
                raise
        raise last_err or RuntimeError("Ollama request failed")

    def _stream_lines(self, path: str, payload: Dict[str, Any]) -> Iterator[str]:
        url = f"{self.base_url}{path}"
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=httpx.Timeout(self.timeout_s)) as c:
                    with c.stream("POST", url, json=payload) as resp:
                        if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                            time.sleep(0.4 * (attempt + 1))
                            continue
                        resp.raise_for_status()
                        for line in resp.iter_lines():
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue
                            # Ollama streaming keys differ between endpoints
                            if "response" in obj and isinstance(obj["response"], str):
                                yield obj["response"]
                            elif "message" in obj and isinstance(obj["message"], dict):
                                chunk = obj["message"].get("content")
                                if isinstance(chunk, str) and chunk:
                                    yield chunk
                            if obj.get("done") is True:
                                return
                return
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(0.4 * (attempt + 1))
                    continue
                raise
        raise last_err or RuntimeError("Ollama streaming failed")

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
    ) -> Union[LLMResponse, Iterator[str]]:
        options: Dict[str, Any] = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "num_predict": int(max_tokens),
        }
        if seed is not None:
            options["seed"] = int(seed)
        if stop:
            options["stop"] = stop

        payload = {"model": self.model, "prompt": prompt, "stream": bool(stream), "options": options}
        if stream:
            return self._stream_lines("/api/generate", payload)

        data = self._request_json("POST", "/api/generate", payload)
        text = (data.get("response") or "").strip()
        return LLMResponse(text=text, raw=data)

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
    ) -> Union[LLMResponse, Iterator[str]]:
        options: Dict[str, Any] = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "num_predict": int(max_tokens),
        }
        if seed is not None:
            options["seed"] = int(seed)

        payload = {
            "model": self.model,
            "messages": list(messages),
            "stream": bool(stream),
            "options": options,
        }
        if stream:
            return self._stream_lines("/api/chat", payload)

        data = self._request_json("POST", "/api/chat", payload)
        msg = data.get("message") or {}
        text = (msg.get("content") or "").strip()
        return LLMResponse(text=text, raw=data)

    def summarize(
        self,
        text: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> Union[LLMResponse, Iterator[str]]:
        prompt = (
            "Summarize the following text clearly and concisely.\n\n"
            "Return markdown with a short summary and bullet points.\n\n"
            f"TEXT:\n{text}"
        )
        return self.generate(prompt, temperature=temperature, max_tokens=max_tokens, stream=stream)

