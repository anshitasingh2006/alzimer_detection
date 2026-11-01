import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAxVaesEB2skVSneYdU17zNFuQNVliRwk8")

try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"⚠️ Warning: Could not configure genai: {e}")


def get_gemini_model():
    """
    Return a configured Gemini model instance.
    If you change model name, change in one place here.
    """
    try:
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        print(f"⚠️ Warning: Could not create Gemini model: {e}")
        return None

