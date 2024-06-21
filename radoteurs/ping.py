from __future__ import annotations

from typing import Optional

from .base import Radoteur
from .messages import Thread


class RadoteurPing(Radoteur):
    """
    Un assistant de test.
    """

    def __init__(self):
        pass

    async def reply(self, thread: Thread) -> Optional[str]:
        if thread.messages[-1].content.endswith("ping"):
            return "pong"
