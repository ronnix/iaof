from __future__ import annotations

from pathlib import Path
import logging
import os

import discord
from dotenv import load_dotenv
from semantic_text_splitter import CharacterTextSplitter

from llm import AOFGPT, Message, Thread


HERE = Path(__file__).parent


DEFAULT_STYLE = "concis, poli, inclusif (un léger grain de poésie est autorisé)"

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
        intents.message_content = True  # il faut pouvoir lire les messages pour y répondre
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
        """
        Récupère les messages précédents de la conversation, pour avoir le contexte
        lorsque lorsque la personne répond au bot.
        """
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


def main() -> None:
    load_dotenv()  # charge les variables d’environnement depuis un fichier .env

    aof_gpt = AOFGPT(
        api_key=os.environ["OPENAI_API_KEY"],
        instructions=(HERE / "instructions.md").read_text(),
        default_style=DEFAULT_STYLE,
    )

    discord_token = os.environ["DISCORD_TOKEN"]
    client = AOFDiscordClient(aof_gpt)
    client.run(discord_token)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
