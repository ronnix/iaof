# Un chatbot pour AOF

Un petit projet pour expérimenter avec les APIs Discord et OpenAI.

## Prérequis

- Python 3.10

## Préparatifs

Côté Discord :
- créer un compte Discord
- créer une application dans https://discord.com/developers/applications
- copier le token dans la section Settings > Bot

Côté OpenAI :
- créer un compte OpenAI
- créer une clé d’API dans https://platform.openai.com/api-keys

## Installation

Dans un environnement virtuel Python :

	pip install -r requirements.txt

## Configuration

Dans un fichier `.env`, définissez les variables d’environnement suivantes :

	DISCORD_TOKEN=<le jeton du bot de l’application discord>
	OPENAI_API_KEY=<la clé d’API OpenAI>

## Lancement

	python bot.py
