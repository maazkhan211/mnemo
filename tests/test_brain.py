"""End-to-end tests using the real embedding model, so they check that matching is semantic, not keyword-based."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import pytest

from brain import Brain
from brain.config import EMBED_MODEL
from brain.enrich import TaggingError, detect_type, normalize_tags, suggest_tags
from brain.vectors import Embedder


@pytest.fixture(scope="session")
def embedder():
    return Embedder(EMBED_MODEL)


@pytest.fixture
def brain(tmp_path, embedder):
    return Brain(data_dir=tmp_path, embedder=embedder, min_connection_score=0.3)


def seed(brain: Brain) -> dict[str, int]:
    now = datetime.now()
    texts = {
        "sleep": "Screens late at night ruin how well I sleep.",
        "coffee": "Drinking caffeine in the afternoon keeps me awake at bedtime.",
        "vectors": "Vector databases answer nearest-neighbour queries over embeddings.",
        "pasta": "Recipe: garlic, olive oil and chilli flakes make a quick pasta sauce.",
    }
    ids = {}
    for i, (key, text) in enumerate(texts.items()):
        note, _ = brain.save_note(text, timestamp=now - timedelta(days=30 - i))
        ids[key] = note.id
    return ids


def test_search_is_semantic(brain):
    ids = seed(brain)
    # Hardly any words shared with the stored notes, but both sleep notes should outrank the rest.
    results = brain.search("insomnia from using my phone in bed", k=2)
    assert {n.id for n, _ in results} == {ids["sleep"], ids["coffee"]}

    results = brain.search("similarity search for ML vectors", k=1)
    assert results[0][0].id == ids["vectors"]


def test_connections_are_made_on_save_and_are_bidirectional(brain):
    ids = seed(brain)
    coffee_links = {n.id for n, _ in brain.connections(ids["coffee"])}
    assert ids["sleep"] in coffee_links
    assert ids["pasta"] not in coffee_links
    # The older note links forward to the newer one too.
    assert ids["coffee"] in {n.id for n, _ in brain.connections(ids["sleep"])}


def test_note_never_connects_to_itself(brain):
    ids = seed(brain)
    for note_id in ids.values():
        assert note_id not in {n.id for n, _ in brain.connections(note_id)}


def test_delete_removes_note_vector_and_connections(brain):
    ids = seed(brain)
    brain.delete_note(ids["sleep"])
    assert brain.db.get_note(ids["sleep"]) is None
    assert brain.vectors.get_embedding(ids["sleep"]) is None
    assert ids["sleep"] not in {n.id for n, _ in brain.connections(ids["coffee"])}
    assert all(r[0].id != ids["sleep"] for r in brain.search("sleep", k=10))


def test_resurface_pairs_old_note_with_recent_related_one(brain):
    old, _ = brain.save_note("Screens late at night ruin my sleep.", timestamp=datetime.now() - timedelta(days=40))
    brain.save_note("Garlic pasta recipe.", timestamp=datetime.now() - timedelta(days=1))
    recent, _ = brain.save_note("Stopped using my phone in bed and I sleep better.")

    pick = brain.resurface(min_age_days=7, recent_days=7, rng=random.Random(0))
    assert pick.old.id == old.id
    assert pick.recent.id == recent.id


def test_resurface_with_no_old_notes(brain):
    brain.save_note("brand new note")
    assert brain.resurface(min_age_days=7) is None


def test_rebuild_restores_index(brain):
    ids = seed(brain)
    before = sorted(brain.db.all_connections())
    brain.vectors.reset()
    assert brain.search("sleep") == []
    brain.rebuild()
    assert brain.search("bad sleep", k=1)[0][0].id == ids["sleep"]
    assert {(a, b) for a, b, _ in brain.db.all_connections()} >= {(a, b) for a, b, _ in before}


def test_graph(brain):
    ids = seed(brain)
    g = brain.graph()
    assert g.number_of_nodes() == 4
    assert g.has_edge(ids["sleep"], ids["coffee"])


def test_tagger_is_used_and_normalised(brain):
    brain.tagger = lambda text: ["#Health", "Sleep Hygiene"]
    note, _ = brain.save_note("Bed at 10.", tags="Habits")
    assert note.tags == ("habits", "health", "sleep-hygiene")
    note, _ = brain.save_note("No tagging please", auto_tag=False)
    assert note.tags == ()


def test_empty_note_rejected(brain):
    with pytest.raises(ValueError):
        brain.save_note("   ")


def test_helpers():
    assert detect_type("check https://example.com later") == "link"
    assert detect_type("just a thought") == "note"
    assert normalize_tags(" ML, #python ,ml,, Deep Learning") == ["ml", "python", "deep-learning"]


def test_suggest_tags_raises_on_unreachable_ollama():
    with pytest.raises(TaggingError):
        suggest_tags("hello", url="http://127.0.0.1:9", timeout=1)


def test_tagger_failure_still_saves_note(brain):
    def broken(text):
        raise TaggingError("boom")

    brain.tagger = broken
    note, _ = brain.save_note("Still saved", tags="manual")
    assert note.tags == ("manual",)
    assert brain.last_tag_error == "boom"
    assert brain.search("still saved", k=1)[0][0].id == note.id


def test_update_note_text_reembeds_and_reconnects(brain):
    ids = seed(brain)
    original = brain.db.get_note(ids["pasta"])

    updated = brain.update_note(ids["pasta"], text="Herbal tea before bed helps me fall asleep.", tags="Sleep, health")
    assert updated.text == "Herbal tea before bed helps me fall asleep."
    assert updated.tags == ("sleep", "health")
    assert updated.timestamp == original.timestamp  # creation time is kept

    # The stored vector, search and connections all reflect the new text, not the old pasta recipe.
    stored = brain.vectors.get_embedding(ids["pasta"])
    expected = brain.embedder.encode([updated.text])[0]
    assert sum(a * b for a, b in zip(stored, expected)) > 0.999
    assert ids["pasta"] in {n.id for n, _ in brain.search("drinks that help me sleep", k=3)}
    assert ids["sleep"] in {n.id for n, _ in brain.connections(ids["pasta"])}
    assert ids["pasta"] in {n.id for n, _ in brain.connections(ids["sleep"])}


def test_update_note_partial_keeps_other_fields(brain):
    note, _ = brain.save_note("Original text", type="idea", tags="a, b")
    updated = brain.update_note(note.id, tags="c")
    assert (updated.text, updated.type, updated.tags) == ("Original text", "idea", ("c",))
    updated = brain.update_note(note.id, type="todo")
    assert (updated.type, updated.tags) == ("todo", ("c",))


def test_update_note_rejects_empty_and_missing(brain):
    note, _ = brain.save_note("Something")
    with pytest.raises(ValueError):
        brain.update_note(note.id, text="   ")
    with pytest.raises(KeyError):
        brain.update_note(9999, text="x")
