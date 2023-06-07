from promptrix.Prompt import Prompt
from promptrix.SystemMessage import SystemMessage
from promptrix.ConversationHistory import ConversationHistory
from promptrix.UserMessage import UserMessage
from promptrix.promptrixTypes import Message
#from alphawave.OpenAIClient  import OpenAIClient
from alphawave.OSClient  import OSClient
from alphawave.AlphaWave import AlphaWave
from alphawave.alphawaveTypes import PromptCompletionOptions
from alphawave.JSONResponseValidator import JSONResponseValidator
import os
import jsonschema
import readline as re
import asyncio

# Read in .env file.

# Create an OpenAI or AzureOpenAI client
client = OSClient(apiKey=os.getenv("OPENAI_API_KEY"))

# Define expected response schema and create a validator
ResponseSchema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]}
    },
    "required": ["answer", "sentiment"]
}

validator = JSONResponseValidator(ResponseSchema)

# Create a wave
wave = AlphaWave(
    client=client,
    prompt=Prompt([
        UserMessage("You are helpful, creative, clever, and very friendly."),
        UserMessage(
"""<INSTRUCTION>: Respond only in JSON using format:
{'answer':'<answer>','sentiment':'positive|neutral|negative'}
Answer the user and analyze the sentiment of user input. </INSTRUCTION>
"""
        ),
        ConversationHistory('history'),
        UserMessage('{{$input}}', 200)
    ]),
    prompt_options=PromptCompletionOptions(
        completion_type="chat",
        model="WIZARDLM_30B",
        temperature=0.9,
        max_input_tokens=2000,
        max_tokens=1000,
    ),
    validator=validator,
    logRepairs=True
)

# Define main chat loop
async def chat(botMessage=None):
    # Show the bots message
    if botMessage:
        if "sentiment" in botMessage:
            print(f"[{botMessage['sentiment']}]")
        print(f"{botMessage['answer']}")

    # Prompt the user for input
    inputtxt = input('User: ')
    # Check if the user wants to exit the chat
    if inputtxt.lower() == 'exit':
        # Close the readline interface and exit the process
        exit()
    else:
        # Route users message to wave
        result = await wave.completePrompt(inputtxt)
        print(result)
        if result['status'] == 'success':
            await chat(result['message']['content'])
        else:
            if result['message']:
                print(f"{result['status']}: {result['message']}")
            else:
                print(f"A result status of '{result['status']}' was returned.")
            exit()

# Start chat session
asyncio.run(chat({"answer": "Hello, how can I help you?"}))
