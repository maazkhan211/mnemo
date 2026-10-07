"""Fill a separate demo brain with backdated sample notes, for trying out search, resurfacing and the graph.

    python scripts/seed_demo.py                     # writes to ./data_demo
    BRAIN_DATA_DIR=data_demo streamlit run app.py   # (PowerShell: $env:BRAIN_DATA_DIR="data_demo")

Your real notes in ./data are never touched unless you pass --data-dir data.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain import Brain  # noqa: E402
from brain.enrich import ollama_available, suggest_tags  # noqa: E402

# (days ago, type, text, tags). Tags are given so seeding works without Ollama.
NOTES = [
    (60, "note", "Read about how transformers use self-attention to weigh every token against every other token.", "machine-learning"),
    (58, "thought", "Slept badly again. Screens after 11pm seem to wreck my sleep quality.", "health,sleep"),
    (55, "idea", "Side project: a CLI that summarises my git commits into a weekly changelog.", "projects,dev-tools"),
    (52, "note", "Vector databases store embeddings and answer nearest-neighbour queries, e.g. Chroma, Qdrant, pgvector.", "machine-learning,databases"),
    (50, "note", "Morning walks in daylight help reset the circadian rhythm.", "health,sleep"),
    (47, "todo", "Ask Sara for feedback on the Q3 roadmap draft before Friday.", "work"),
    (45, "note", "Cosine similarity measures the angle between two vectors and ignores their length.", "machine-learning,math"),
    (41, "thought", "I learn best when I write things down in my own words, not by highlighting.", "learning"),
    (38, "note", "Spaced repetition: reviewing material at growing intervals beats cramming.", "learning"),
    (35, "idea", "Build a personal search engine over my notes using local embeddings.", "projects,machine-learning"),
    (32, "note", "Python's functools.lru_cache memoises pure functions with almost no code.", "python"),
    (30, "note", "Magnesium glycinate before bed is often recommended for sleep, though evidence is mixed.", "health,sleep"),
    (26, "thought", "Meetings without an agenda nearly always run over and decide nothing.", "work,productivity"),
    (21, "note", "Retrieval-augmented generation: fetch relevant documents, then feed them to the LLM as context.", "machine-learning,llm"),
    (18, "note", "SQLite handles far more than people expect. WAL mode allows concurrent readers.", "databases"),
    (14, "idea", "Weekly review ritual: skim the past week's notes every Sunday and link related ones.", "productivity,learning"),
    (10, "note", "Streamlit reruns the whole script on every interaction; cache expensive resources.", "python,dev-tools"),
    (6, "thought", "Caffeine after 2pm keeps me up even when I feel tired.", "health,sleep"),
    (5, "note", "sentence-transformers all-MiniLM-L6-v2 gives 384-dim embeddings and runs fine on CPU.", "machine-learning"),
    (3, "todo", "Prepare a lightning talk on semantic search for the team meetup.", "work,machine-learning"),
    (2, "note", "Time-blocking my calendar made deep work much easier to protect.", "productivity"),
    (1, "thought", "Notes become far more valuable once they're connected to each other, not just stored.", "learning,projects"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data_demo", help="where to write the demo brain (default: data_demo)")
    parser.add_argument("--llm-tags", action="store_true", help="also ask Ollama for tags")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if data_dir.resolve() == (Path(__file__).resolve().parent.parent / "data").resolve():
        if input(f"This adds demo notes to your REAL brain at {data_dir}. Continue? [y/N] ").lower() != "y":
            return
    elif data_dir.exists():
        shutil.rmtree(data_dir)

    use_llm = args.llm_tags and ollama_available()
    brain = Brain(data_dir=data_dir, tagger=suggest_tags if use_llm else None)
    now = datetime.now()
    for days_ago, note_type, text, tags in NOTES:
        note, related = brain.save_note(text, type=note_type, tags=tags, auto_tag=use_llm,
                                        timestamp=now - timedelta(days=days_ago, hours=days_ago % 7))
        print(f"#{note.id:>2} {note.title[:50]:<50} -> {len(related)} connections")

    s = brain.stats()
    print(f"\nDone: {s['notes']} notes, {s['connections']} connections, {s['tags']} tags in {data_dir}")


if __name__ == "__main__":
    main()
