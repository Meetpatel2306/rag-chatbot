from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# Get API key from .env file (Safe way)
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ ERROR: GROQ_API_KEY not found in .env file.")
else:
    client = Groq(api_key=api_key)

    # Simplified test call
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Hello! Test connection."}],
        )
        print("✅ SUCCESS:", response.choices[0].message.content)
    except Exception as e:
        print("❌ FAILED:", str(e))
