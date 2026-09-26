"""InspectIQ — inspection compliance copilot."""
from pathlib import Path

from dotenv import load_dotenv

# Load .env as soon as the package is imported, before any third-party library reads its
# environment variables (huggingface_hub reads HF_HUB_OFFLINE once, at import time).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
