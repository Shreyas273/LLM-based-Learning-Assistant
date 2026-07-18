from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Protocol, Sequence, TypedDict, Union


class ChatMessage(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    text: str
    raw: Optional[Dict[str, Any]] = None


class BaseLLMProvider(ABC):
    """
    Abstract LLM provider interface.

    Providers SHOULD be:
    - deterministic by default for RAG (low temperature)
    - safe with timeouts/retries
    - able to return raw payload for debugging
    """

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def summarize(
        self,
        text: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> Union[LLMResponse, Iterator[str]]:
        raise NotImplementedError

