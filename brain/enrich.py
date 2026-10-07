"""Optional enrichment of notes: auto-tagging with a local LLM (Ollama) and fetching titles for bare links."""

from __future__ import annotations

import html
import json
import re

import requests

from brain.config import OLLAMA_MODEL, OLLAMA_URL

URL_RE = re.compile(r"https?://\S+")
TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)

TAG_PROMPT = """You label personal notes with short topic tags.
Return 1 to 3 tags for the note below. Tags are lowercase, one or two words, general enough
to be reused across many notes (e.g. "machine-learning", "health", "career", "python").
Respond only with JSON: {{"tags": ["tag1", "tag2"]}}

Note:
\"\"\"{text}\"\"\""""


def normalize_tags(tags) -> list[str]:
    """Lowercase, strip '#', hyphenate spaces, drop empties and duplicates (order preserved)."""
    if isinstance(tags, str):
        tags = tags.split(",")
    seen: list[str] = []
    for tag in tags or []:
        if not isinstance(tag, str):
            continue
        tag = re.sub(r"\s+", "-", tag.strip().lstrip("#").strip().lower())
        tag = re.sub(r"[^a-z0-9\-_./+]", "", tag)[:30]
        if tag and tag not in seen:
            seen.append(tag)
    return seen


def detect_type(text: str) -> str:
    return "link" if URL_RE.search(text) else "note"


def ollama_available(url: str = OLLAMA_URL, model: str = OLLAMA_MODEL, timeout: float = 2.0) -> bool:
    """True if Ollama is running and the configured model is pulled."""
    try:
        resp = requests.get(f"{url}/api/tags", timeout=timeout)
        resp.raise_for_status()
        names = {m.get("name", "") for m in resp.json().get("models", [])}
    except (requests.RequestException, ValueError):
        return False
    return model in names or f"{model}:latest" in names


class TaggingError(RuntimeError):
    pass


def suggest_tags(text: str, url: str = OLLAMA_URL, model: str = OLLAMA_MODEL, timeout: float = 60.0) -> list[str]:
    """Ask the local LLM for 1-3 tags. Raises TaggingError on failure (Brain.save_note catches it)."""
    try:
        resp = requests.post(
            f"{url}/api/generate",
            json={
                "model": model,
                "prompt": TAG_PROMPT.format(text=text[:2000]),
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=timeout,
        )
        body = resp.json()
        if "error" in body:
            raise TaggingError(f"Ollama error: {body['error']}")
        resp.raise_for_status()
        tags = json.loads(body["response"]).get("tags", [])
    except requests.RequestException as e:
        raise TaggingError(f"Could not reach Ollama at {url}: {e}") from e
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise TaggingError(f"Unexpected reply from {model}: {e}") from e
    return normalize_tags(tags)[:3]


def fetch_link_title(url: str, timeout: float = 5.0) -> str | None:
    """Fetch a page's <title>. A bare URL embeds poorly; its title gives search something to match."""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 SecondBrain"})
        resp.raise_for_status()
    except requests.RequestException:
        return None
    match = TITLE_RE.search(resp.text[:200_000])
    if not match:
        return None
    title = html.unescape(re.sub(r"\s+", " ", match.group(1))).strip()
    return title[:200] or None


def enrich_bare_link(text: str) -> str:
    """If the note is nothing but a URL, prepend the page title. Notes with any other text are untouched."""
    stripped = text.strip()
    if not URL_RE.fullmatch(stripped):
        return text
    title = fetch_link_title(stripped)
    return f"{title}\n{stripped}" if title else text
