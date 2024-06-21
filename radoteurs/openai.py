from __future__ import annotations

from dataclasses import asdict
from typing import Optional
import json

from openai import AsyncOpenAI

from .base import Radoteur
from .messages import Message, Thread


# On utilise le modèle GPT 4 Turbo, qui est actuellement le plus performant.
DEFAULT_MODEL = "gpt-4-0125-preview"


class RadoteurOpenAI(Radoteur):
    """
    Un assistant basé sur l’API Chat d’OpenAI.

    Une version expérimentale basée sur l’API Assistant (beta) existe
    sur une autre branche.
    """

    provider = "OpenAI"

    def __init__(
        self,
        api_key: str,
        instructions: str,
        default_style: str,
        model_name=DEFAULT_MODEL,
    ) -> None:
        super().__init__(instructions, default_style)
        self.openai_client = AsyncOpenAI(api_key=api_key)
        self.model_name = model_name

    async def reply(self, thread: Thread) -> Optional[str]:
        # On va passer le prompt système + le thread de messages entre l’utilisateur et le bot
        messages = [
            Message(role="system", content=self.system_prompt(thread.context))
        ] + thread.messages

        # On appelle l’API
        completion = await self.openai_client.chat.completions.create(
            messages=[asdict(message) for message in messages],
            model=self.model_name,
            max_tokens=1024,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "changer_le_style",
                        "description": "Permet de changer le style, la manière de parler de IAOF.",
                        "parameters": {
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
                },
            ],
        )

        # Quelque chose n’a pas fonctionné
        if completion is None:
            return None

        # Cas d’une réponse textuelle normale
        message = completion.choices[0].message
        if message.content is not None:
            return message.content

        # Cas où le modèle choisit d’utiliser un des outils mis à sa disposition
        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.type == "function":
                    if tool_call.function.name == "changer_le_style":
                        args = json.loads(tool_call.function.arguments)
                        return self.changer_le_style(
                            style=args["style"],
                            context=thread.context,
                        )
