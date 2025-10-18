from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env from project root regardless of where this script is run
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

key = os.getenv("OPENAI_API_KEY", "")
print("Key present:", bool(key))
print("Starts with sk-:", key.startswith("sk-"))
