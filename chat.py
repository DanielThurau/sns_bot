from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=OPENAI_API_KEY
)


chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say hello Daniel back to me",
        }
    ],
    model="gpt-3.5-turbo",
)
print(chat_completion.choices[0].message.content.strip())