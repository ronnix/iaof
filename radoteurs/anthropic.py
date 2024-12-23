from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from anthropic import AsyncAnthropic

from .base import Radoteur
from .messages import Thread


# On utilise le modèle Claude 3.5 Sonnet, qui est actuellement le plus performant.
DEFAULT_MODEL = "claude-3-5-sonnet-latest"


class RadoteurAnthropic(Radoteur):
    """
    Un assistant basé sur l’API Claude d’Anthropic.
    """

    provider = "Claude d’Anthropic"

    def __init__(
        self,
        api_key: str,
        instructions: str,
        default_style: str,
        model_name=DEFAULT_MODEL,
    ) -> None:
        super().__init__(instructions, default_style)
        self.client = AsyncAnthropic(api_key=api_key)
        self.model_name = model_name

    async def reply(self, thread: Thread) -> Optional[str]:
        # On appelle l’API
        message = await self.client.messages.create(
            messages=[asdict(message) for message in thread.messages],
            system=self.system_prompt(thread.context),
            model=self.model_name,
            max_tokens=1024,
            tools=[
                {
                    "name": "changer_le_style",
                    "description": "Permet de changer le style, la manière de parler de IAOF. À utiliser seulement si l’utilisateur te demande de changer de manière de parler, et si le style actuel n’a pas déjà la valeur requise.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "style": {
                                "type": "string",
                                "description": "Une description du style souhaité (p. ex. « \u00e0 la mani\u00e8re de Pierre Desproges »)",
                            },
                        },
                        "required": ["style"],
                    },
                },
            ],
        )

        # Quelque chose n’a pas fonctionné
        if message is None:
            return None

        # On récupère le contenu des blocs de texte
        result = [block.text for block in message.content if block.type == "text"]

        # Le modèle a-t’il choisi d’utiliser un outil mis à sa disposition ?
        if message.stop_reason == "tool_use":
            for block in message.content:
                if block.type == "tool_use":
                    if block.name == "changer_le_style":
                        result.append(
                            self.changer_le_style(
                                style=block.input["style"],
                                context=thread.context,
                            )
                        )

        return "\n\n".join(result)
