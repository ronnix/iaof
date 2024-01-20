from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from string import Template
from typing import Literal, Optional
import codecs
import json
import logging
import os
import re

import discord
from dotenv import load_dotenv
from ftfy import fix_text
from openai import AsyncOpenAI
from semantic_text_splitter import CharacterTextSplitter


HERE = Path(__file__).parent


MAX_MESSAGE_SIZE = 2_000


class AOFDiscordClient(discord.Client):
    """
    Le bot Discord.
    """

    def __init__(self, aof_gpt: AOFGPT) -> None:
        super().__init__(intents=self._intents())
        self.aof_gpt = aof_gpt

    def _intents(self) -> discord.Intents:
        intents = discord.Intents.default()
        intents.message_content = True
        return intents

    async def on_ready(self) -> None:
        logging.info(f"{self.user} s’est connecté à Discord")

    async def on_message(self, message: discord.Message) -> None:
        # On ne se répond pas à soi-même
        if message.author == self.user:
            return

        # On réagit seulement à une mention ou à un message privé
        is_mention = self.user.id in {member.id for member in message.mentions}
        is_dm = message.channel.type == discord.ChannelType.private
        if not (is_mention or is_dm):
            return

        # On signale qu’on va répondre
        async with message.channel.typing():
            # On demande au LLM de produire une réponse
            try:
                thread = await self.make_thread(message)
                response = await self.aof_gpt.reply(thread)
            except:
                logging.exception("Erreur en générant la réponse")
                response = None
            if response is None:
                response = "Oups, une erreur a eu lieu…"

            # On poste la réponse, en la découpant si elle est trop longue pour un seul message
            try:
                for chunk in chunked(response, MAX_MESSAGE_SIZE):
                    await message.reply(chunk)
            except:
                logging.exception("Erreur en postant la réponse")

    async def make_thread(self, message) -> Thread:
        if message.channel.type == discord.ChannelType.private:
            context = f"user-{message.author.id}"
        else:
            assert message.guild is not None
            context = f"server-{message.guild.id}"

        messages = []
        while message:
            messages.insert(
                0,
                Message(
                    role="assistant" if message.author.id == self.user.id else "user",
                    content=message.clean_content,
                ),
            )
            if reference := message.reference:
                if reference.cached_message:
                    message = reference.cached_message
                elif reference.resolved:
                    message = reference.resolved
                else:
                    message = await message.channel.fetch_message(reference.message_id)
            else:
                message = None
        return Thread(context=context, messages=messages)


def chunked(text: str, max_size: int) -> list[str]:
    """
    Découpe en morceaux un message trop long, selon des frontières de mots.
    """
    splitter = CharacterTextSplitter()
    return splitter.chunks(text, max_size)


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


Context = str


@dataclass
class Thread:
    context: Context
    messages: list[Message]


class AOFGPT:
    """
    L’assistant, basé sur ChatGPT.
    """

    def __init__(
        self,
        api_key: str,
        instructions: str,
        default_style: str,
        model_name="gpt-4-1106-preview",
    ) -> None:
        self.openai_client = AsyncOpenAI(api_key=api_key)
        self.instructions = Template(instructions)
        self.model_name = model_name
        self.default_style = default_style
        self.styles: dict[Context, str] = {}
        self.styles_path = Path("styles.json")
        if self.styles_path.exists():
            self.styles = json.load(self.styles_path.open())
        logging.info("Styles : %s", self.styles)

    def system_prompt(self, context: Context) -> str:
        return self.instructions.safe_substitute(
            model_name=self.model_name,
            style=self.styles.get(context, self.default_style),
        )

    async def reply(self, thread: Thread) -> Optional[str]:
        messages = [
            Message(role="system", content=self.system_prompt(thread.context))
        ] + thread.messages
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
        if completion is None:
            return None

        message = completion.choices[0].message
        if message.content is not None:
            return message.content

        if message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.type == "function":
                    if tool_call.function.name == "changer_le_style":
                        args = json.loads(tool_call.function.arguments)
                        return self.changer_le_style(
                            style=clean_text(args["style"]),
                            context=thread.context,
                        )

    def changer_le_style(self, style: str, context: Context) -> str:
        self.styles[context] = style
        json.dump(self.styles, self.styles_path.open("w"))
        return f"Ok, mon style est maintenant « {style} »."


def clean_text(s: str) -> str:
    """
    L’API OpenAI a tendance à mal encoder les caractères accentués dans les appels d’outils
    """
    if r"\u" in s:
        s = codecs.decode(s, "unicode_escape")
    s = fix_text(s)
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    load_dotenv()

    aof_gpt = AOFGPT(
        api_key=os.environ["OPENAI_API_KEY"],
        instructions=(HERE / "instructions.md").read_text(),
        default_style="concis, poli, inclusif (un léger grain de poésie est autorisé)",
    )

    discord_token = os.environ["DISCORD_TOKEN"]
    client = AOFDiscordClient(aof_gpt)
    client.run(discord_token)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
