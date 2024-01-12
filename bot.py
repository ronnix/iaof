from __future__ import annotations

from pathlib import Path
from typing import Optional
import asyncio
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
        print(f"{self.user} s’est connecté à Discord")
        await self.aof_gpt.create_or_update()

    async def on_message(self, message: discord.Message) -> None:
        # On ne se répond pas à soi-même
        if message.author == self.user:
            return

        # On réagit seulement à une mention ou à un message privé
        if self.user.id not in {member.id for member in message.mentions}:
            return

        # On signale qu’on va répondre
        async with message.channel.typing():
            # On demande au LLM de produire une réponse
            replies = await self.aof_gpt.replies(message)
            if not replies:
                replies = ["Oups, une erreur a eu lieu…"]

            # On poste les réponses, en les découpant si elles sont trop longues pour un seul message
            for reply in replies:
                for chunk in chunked(reply, MAX_MESSAGE_SIZE):
                    await message.reply(chunk)


def chunked(text: str, max_size: int) -> list[str]:
    """
    Découpe en morceaux un message trop long, selon des frontières de mots.
    """
    splitter = CharacterTextSplitter()
    return splitter.chunks(text, max_size)


class AOFGPT:
    """
    L’assistant, basé sur ChatGPT.
    """

    name = "IAOF"

    def __init__(
        self,
        api_key: str,
        instructions: str,
        assistant_id: Optional[str] = None,
        model="gpt-4-1106-preview",
    ) -> None:
        self.openai_client = AsyncOpenAI(api_key=api_key)
        self.instructions = instructions
        self.model = model
        self.assistant_id = assistant_id
        self.assistant = None
        self.threads = {}

    async def create_or_update(self):
        if self.assistant_id is None:
            self.assistant = await self.create_assistant()
            self.assistant_id = self.assistant.id
            print(f"ASSISTANT_ID={self.assistant_id}")
        else:
            self.assistant = await self.retrieve_and_update_assistant(self.assistant_id)
        print("L’assistant est prêt.")

    async def create_assistant(self):
        print("Création de l’assistant…")
        assistant = await self.openai_client.beta.assistants.create(
            name=self.name,
            instructions=self.instructions,
            model=self.model,
        )
        return assistant

    async def retrieve_and_update_assistant(self, assistant_id: str):
        print("Recherche de l’assistant…")
        assistant = await self.openai_client.beta.assistants.retrieve(
            assistant_id=assistant_id
        )

        print("Mise à jour de l’assistant…")
        assistant = await self.openai_client.beta.assistants.update(
            assistant_id,
            name=self.name,
            instructions=self.instructions,
            model=self.model,
        )
        return assistant

    async def get_or_create_thread(self, discord_message):
        thread = await self.openai_client.beta.threads.create()
        message = await self.openai_client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=discord_message.clean_content,
        )
        return thread, message

    async def replies(self, discord_message) -> list[str]:
        assert self.assistant is not None

        thread, message = await self.get_or_create_thread(discord_message)

        run = await self.openai_client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.assistant.id,
        )

        await self.wait_on_run(run, thread)

        messages = await self.openai_client.beta.threads.messages.list(
            thread_id=thread.id,
            order="asc",
            after=message.id,
        )
        return [
            content.text.value
            for message in messages.data
            for content in message.content
            if content.type == "text"
        ]

    async def wait_on_run(self, run, thread):
        while run.status == "queued" or run.status == "in_progress":
            await asyncio.sleep(0.5)
            run = await self.openai_client.beta.threads.runs.retrieve(
                run_id=run.id,
                thread_id=thread.id,
            )


def main() -> None:
    load_dotenv()

    aof_gpt = AOFGPT(
        api_key=os.environ["OPENAI_API_KEY"],
        instructions=(HERE / "instructions.md").read_text(),
        assistant_id=os.getenv("ASSISTANT_ID"),
    )

    discord_token = os.environ["DISCORD_TOKEN"]
    client = AOFDiscordClient(aof_gpt)
    client.run(discord_token)


if __name__ == "__main__":
    main()
