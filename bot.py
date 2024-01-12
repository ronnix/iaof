from pprint import pprint
import os

import discord
from dotenv import load_dotenv


class AOFDiscordClient(discord.Client):
    def __init__(self) -> None:
        super().__init__(intents=self._intents())

    def _intents(self) -> discord.Intents:
        intents = discord.Intents.default()
        intents.message_content = True
        return intents

    async def on_ready(self) -> None:
        print(f"{self.user} s’est connecté à Discord")

    async def on_message(self, message: discord.Message) -> None:
        if message.author == self.user:
            return

        if message.content == "ping":
            await message.channel.send("pong", reference=message)


def main() -> None:
    load_dotenv()

    discord_token = os.environ["DISCORD_TOKEN"]

    client = AOFDiscordClient()
    client.run(discord_token)


if __name__ == "__main__":
    main()
