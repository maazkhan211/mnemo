"""Runtime configuration. Every value can be overridden with an environment variable."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.getenv("BRAIN_DATA_DIR", PROJECT_ROOT / "data"))
EMBED_MODEL = os.getenv("BRAIN_EMBED_MODEL", "all-MiniLM-L6-v2")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# How many auto-connections to store per new note, and the minimum cosine
# similarity for two notes to count as connected.
MAX_CONNECTIONS = int(os.getenv("BRAIN_MAX_CONNECTIONS", "5"))
MIN_CONNECTION_SCORE = float(os.getenv("BRAIN_MIN_CONNECTION_SCORE", "0.35"))
