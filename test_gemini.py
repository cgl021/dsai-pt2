import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
print("API key loaded?", api_key is not None)

genai.configure(api_key=api_key)

print("\nAvailable Gemini models:")
for m in genai.list_models():
    # Only print Gemini text models
    if "gemini" in m.name:
        print("-", m.name)
