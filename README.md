# 🧠 Mnemo: your local AI second brain

> *Named after Mnemosyne, the Greek goddess of memory.*

Dump notes, links and thoughts into it every day. It embeds each note locally, connects it to related
notes you wrote before, and brings old notes back when they matter again: when you search, when you
describe what you're working on, or at random in a "Remember this?" card.

Everything runs on your machine. Nothing is sent to a cloud API.

<!-- Add screenshots after a week or two of use:
![Graph view](docs/graph.png)
![Search](docs/search.gif)
-->

## Features

| | |
|---|---|
| ✍️ **Capture** | One text box with Markdown support. The type (note / link / thought / idea / todo) is detected automatically. A bare URL gets its page title fetched so it becomes searchable. |
| 🔗 **Auto-connections** | On save, the note is compared with every existing note. The closest matches (up to 5, cosine similarity ≥ 0.35) are stored as connections in both directions. |
| 🔎 **Semantic search** | Search by meaning, not keywords. Ask "insomnia from my phone" and you'll find "screens late at night ruin my sleep". You can also paste a paragraph about what you're working on. |
| 🏷️ **Auto-tagging** | Optional. A local LLM via [Ollama](https://ollama.com) suggests 1–3 tags. Your own tags are always kept. |
| 💭 **Resurface** | On every app load: a random note older than a week, paired with the most similar recent note. |
| ✏️ **Edit** | Every note has **Edit** and **Delete** buttons. Edit the text, type or tags in place. Changing the text re-embeds the note and recomputes its connections. The original date is kept. |
| 🗂️ **Browse** | Filter by tag, type and text. |
| 🕸️ **Graph** | Interactive network of notes and their connections, colored by tag or type. |

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the embedding model (`all-MiniLM-L6-v2`, ~90 MB). After that the app works offline.

**Optional auto-tagging:** install [Ollama](https://ollama.com), then run `ollama pull llama3.2:3b`. The sidebar
shows whether it's reachable. Without Ollama, everything else works and notes are saved untagged.

### Try it with demo data

```bash
python scripts/seed_demo.py              # 22 backdated sample notes in ./data_demo
```
```powershell
$env:BRAIN_DATA_DIR="data_demo"; streamlit run app.py      # PowerShell
```
```bash
BRAIN_DATA_DIR=data_demo streamlit run app.py              # bash
```

Your real notes live in `./data` and are never touched by the demo.

### Docker

```bash
docker compose up --build     # http://localhost:8501, notes persist in ./data
```

The container reaches Ollama on the host through `host.docker.internal`.

## How it works

```
             ┌─────────────── save_note ────────────────┐
  text ──►  tags (Ollama, optional)                      │
            │                                            ▼
            ├─► SQLite  notes(id, text, timestamp, tags, type)
            │
            ├─► embed (sentence-transformers, 384-dim, normalised)
            │      │
            │      ├─► ChromaDB query: top-5 nearest existing notes
            │      │      └─► SQLite connections(note_id, related_note_id, similarity_score)
            │      └─► ChromaDB upsert (id = note id, metadata = created_at)
```

- **Search:** embed the query → ChromaDB nearest neighbours (cosine) → fetch full notes from SQLite.
- **Resurface:** pick a random note older than N days, then query ChromaDB filtered to `created_at >= now - 7d`.
  If nothing recent is related, fall back to the most similar note of any age.
- **Graph:** `networkx` graph from the connections table, rendered with `pyvis`.

SQLite is the source of truth. The vector index can always be rebuilt from it with
**Sidebar → Maintenance → Rebuild index & connections**. Use this after changing the embedding model or
the connection settings.

### Project layout

```
app.py              Streamlit UI (Capture / Search / Browse / Graph + resurface card)
brain/core.py       Brain: save, search, connections, resurface, graph, rebuild
brain/db.py         SQLite schema and queries
brain/vectors.py    Embedder (sentence-transformers) + VectorStore (ChromaDB)
brain/enrich.py     Ollama tagging, link-title fetching, tag normalisation
scripts/seed_demo.py
tests/test_brain.py End-to-end tests with the real model (checks matching is semantic)
```

## Configuration

All settings are environment variables:

| Variable | Default | |
|---|---|---|
| `BRAIN_DATA_DIR` | `./data` | Where SQLite + ChromaDB live |
| `BRAIN_EMBED_MODEL` | `all-MiniLM-L6-v2` | Any sentence-transformers model. Rebuild the index after changing it. |
| `BRAIN_MAX_CONNECTIONS` | `5` | Connections stored per new note |
| `BRAIN_MIN_CONNECTION_SCORE` | `0.35` | Minimum cosine similarity for a connection |
| `OLLAMA_URL` | `http://localhost:11434` | |
| `OLLAMA_MODEL` | `llama3.2:3b` | |

**Tuning tip:** MiniLM gives fairly low scores between short notes. Unrelated notes usually score 0.1–0.3
and clearly related ones 0.35–0.6. If your graph feels too sparse, try `BRAIN_MIN_CONNECTION_SCORE=0.3`,
or switch to a stronger model such as `BAAI/bge-small-en-v1.5`. Then press Rebuild.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Daily-use checklist (step 13)

- Add notes every day for 1–2 weeks. Connections and resurfacing get better as the brain grows.
- Search using *different words* from the ones in your notes to confirm the matching is semantic.
- Check that the graph forms sensible clusters (health, work, a project...). If unrelated notes are linked,
  raise the threshold. If related notes aren't linked, lower it.
