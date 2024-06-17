from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


Context = str


@dataclass
class Thread:
    context: Context
    messages: list[Message]
