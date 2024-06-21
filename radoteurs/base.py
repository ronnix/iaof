from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Optional
import json
import logging

from babel.dates import format_date, format_time
from pytz import timezone

from .messages import Context, Thread


TZ = timezone("Europe/Paris")


class Radoteur(ABC):
    def __init__(
        self,
        instructions: str,
        default_style: str,
    ) -> None:
        self.instructions = Template(instructions)
        self.init_styles(default_style)

    def init_styles(self, default_style: str) -> None:
        self.default_style = default_style
        self.styles: dict[Context, str] = {}
        self.styles_path = Path("styles.json")
        if self.styles_path.exists():
            self.styles = json.load(self.styles_path.open())
        logging.info("Styles : %s", self.styles)

    def system_prompt(self, context: Context) -> str:
        # Le "system prompt" incorpore quelques éléments dynamiques
        now = datetime.now(tz=TZ)
        return self.instructions.safe_substitute(
            provider=self.provider,
            model_name=self.model_name,
            date=format_date(now, format="long", locale="fr"),
            time=format_time(now, format="short", locale="fr"),
            style=self.styles.get(context, self.default_style),
        )

    @abstractmethod
    async def reply(self, thread: Thread) -> Optional[str]:
        ...

    def changer_le_style(self, style: str, context: Context) -> str:
        """
        Mettre à jour le style souhaité pour le contexte actuel.

        Les consignes de style sont stockées dans un fichier JSON
        de manière à persister en cas de redémarrage du bot.
        """
        self.styles[context] = style
        json.dump(self.styles, self.styles_path.open("w"))
        return f"Ok, mon style est maintenant « {style} »."
