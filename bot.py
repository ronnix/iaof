from pprint import pprint
import os

import discord
from dotenv import load_dotenv


load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")


class AOFDiscordClient(discord.Client):
    async def on_ready(self):
        print(f"{self.user} s’est connecté à Discord")

    async def on_message(self, message):
        if message.author == self.user:
            return

        if message.content == "ping":
            await message.channel.send("pong", reference=message)


def main():
    intents = discord.Intents.default()
    intents.message_content = True

    client = AOFDiscordClient(intents=intents)
    client.run(TOKEN)


if __name__ == "__main__":
    main()
