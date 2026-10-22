
import os
from dotenv import load_dotenv
from groq import Groq

# Load API keys from .env
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY was not found. Check your .env file."
    )

# Create Groq client
client = Groq(api_key=api_key)

# Send a test prompt
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "Explain RAG in simple terms in 3 sentences."
        }
    ],
    temperature=0.3,
)

print("\nGroq response:\n")
print(response.choices[0].message.content)
