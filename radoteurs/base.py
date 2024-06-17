from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .messages import Thread


class Radoteur(ABC):
    @abstractmethod
    async def reply(self, thread: Thread) -> Optional[str]:
        ...
