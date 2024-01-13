from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional
import logging
import os

import discord
from dotenv import load_dotenv
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
        return Thread(messages=messages)


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


@dataclass
class Thread:
    messages: list[Message]


class AOFGPT:
    """
    L’assistant, basé sur ChatGPT.
    """

    def __init__(
        self, api_key: str, system_prompt: str, model_name="gpt-4-1106-preview"
    ) -> None:
        self.openai_client = AsyncOpenAI(api_key=api_key)
        self.system_prompt = system_prompt
        self.model_name = model_name

    async def reply(self, thread: Thread) -> Optional[str]:
        messages = [
            Message(role="system", content=self.system_prompt)
        ] + thread.messages
        completion = await self.openai_client.chat.completions.create(
            messages=[asdict(message) for message in messages],
            model=self.model_name,
            max_tokens=1024,
        )
        if completion is None:
            return None
        return completion.choices[0].message.content


def main() -> None:
    load_dotenv()

    aof_gpt = AOFGPT(
        api_key=os.environ["OPENAI_API_KEY"],
        system_prompt=(HERE / "instructions.md").read_text(),
    )

    discord_token = os.environ["DISCORD_TOKEN"]
    client = AOFDiscordClient(aof_gpt)
    client.run(discord_token)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
