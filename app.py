"""Second Brain: Streamlit interface.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st
from pyvis.network import Network

from brain import Brain, Note
from brain.config import DATA_DIR, MIN_CONNECTION_SCORE, OLLAMA_MODEL
from brain.enrich import enrich_bare_link, ollama_available, suggest_tags

st.set_page_config(page_title="Second Brain", page_icon="🧠", layout="wide")

TYPES = ["note", "thought", "idea", "link", "todo"]
PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
           "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]


@st.cache_resource(show_spinner="Loading embedding model...")
def get_brain() -> Brain:
    return Brain(tagger=suggest_tags, link_enricher=enrich_bare_link)


@st.cache_data(ttl=60, show_spinner=False)
def llm_ready() -> bool:
    return ollama_available()


brain = get_brain()


# --- helpers -----------------------------------------------------------------

def strip_markup(text: str) -> str:
    """Graph tooltips are rendered as HTML, and link notes contain fetched web-page titles."""
    return text.replace("<", "‹").replace(">", "›")


def friendly_time(note: Note) -> str:
    delta = datetime.now() - note.created
    if delta.days == 0:
        return "today " + note.created.strftime("%H:%M")
    if delta.days == 1:
        return "yesterday"
    if delta.days < 30:
        return f"{delta.days} days ago"
    return note.created.strftime("%d %b %Y")


def meta_line(note: Note, score: float | None = None) -> str:
    parts = [f"#{note.id}", friendly_time(note), note.type]
    if note.tags:
        parts.append(" ".join(f"`{t}`" for t in note.tags))
    if score is not None:
        parts.append(f"match **{score:.0%}**")
    return " · ".join(parts)


def render_edit_form(note: Note, key: str) -> None:
    with st.form(f"edit-{key}-{note.id}"):
        text = st.text_area("Note", value=note.text, height=160)
        c1, c2 = st.columns([1, 3])
        types = TYPES if note.type in TYPES else TYPES + [note.type]
        new_type = c1.selectbox("Type", types, index=types.index(note.type))
        tags = c2.text_input("Tags (comma-separated)", value=", ".join(note.tags))
        b1, b2, _ = st.columns([1, 1, 6])
        save = b1.form_submit_button("Save", type="primary")
        cancel = b2.form_submit_button("Cancel")

    if save:
        if not text.strip():
            st.warning("A note can't be empty. Use Delete to remove it instead.")
            return
        brain.update_note(note.id, text=text, type=new_type, tags=tags)
        st.session_state.editing = None
        st.session_state.pop("last_saved", None)  # may show the pre-edit version
        st.toast(f"Note #{note.id} updated")
        st.rerun()
    if cancel:
        st.session_state.editing = None
        st.rerun()


def render_note(note: Note, score: float | None = None, key: str = "") -> None:
    with st.container(border=True):
        if st.session_state.get("editing") == (key, note.id):
            render_edit_form(note, key)
            return
        st.markdown(note.text)
        st.caption(meta_line(note, score))
        related = brain.connections(note.id)
        if related:
            with st.expander(f"🔗 {len(related)} connected note{'s' if len(related) != 1 else ''}"):
                for other, s in related:
                    st.markdown(f"- **{s:.0%}** · {other.title}  \n  <small>#{other.id} · {friendly_time(other)}</small>",
                                unsafe_allow_html=True)
        b1, b2, _ = st.columns([1, 1, 10])
        if b1.button("Edit", key=f"edit-btn-{key}-{note.id}", type="tertiary"):
            st.session_state.editing = (key, note.id)
            st.rerun()
        if b2.button("Delete", key=f"del-{key}-{note.id}", type="tertiary"):
            brain.delete_note(note.id)
            st.session_state.pop("last_saved", None)
            st.rerun()


# --- sidebar -----------------------------------------------------------------

with st.sidebar:
    st.title("🧠 Second Brain")
    stats = brain.stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Notes", stats["notes"])
    c2.metric("Links", stats["connections"])
    c3.metric("Tags", stats["tags"])

    st.divider()
    llm_ok = llm_ready()
    auto_tag = st.toggle("Auto-tag with local LLM", value=llm_ok, disabled=not llm_ok)
    if llm_ok:
        st.caption(f"Ollama ✅ `{OLLAMA_MODEL}`")
    else:
        st.caption(f"Ollama not reachable or `{OLLAMA_MODEL}` not pulled, so auto-tagging is off. "
                   f"Run `ollama pull {OLLAMA_MODEL}`.")

    st.divider()
    with st.expander("Maintenance"):
        st.caption(f"Data folder: `{DATA_DIR}`")
        if st.button("Rebuild index & connections"):
            with st.spinner("Re-embedding all notes..."):
                brain.rebuild()
            st.success("Rebuilt.")


# --- resurface: "remember this?" ---------------------------------------------

if "resurfaced" not in st.session_state:
    st.session_state.resurfaced = brain.resurface()
    st.session_state.resurface_dismissed = False

pick = st.session_state.resurfaced
if pick and not st.session_state.resurface_dismissed:
    with st.container(border=True):
        st.markdown("#### 💭 Remember this?")
        left, right = st.columns(2)
        with left:
            st.caption(f"From {friendly_time(pick.old)}")
            st.markdown(pick.old.text)
        with right:
            if pick.recent:
                st.caption(f"Related to: {friendly_time(pick.recent)} · {pick.score:.0%} match")
                st.markdown(pick.recent.text)
            else:
                st.caption("Nothing related yet. Maybe worth revisiting?")
        b1, b2, _ = st.columns([1, 1, 6])
        if b1.button("Show another"):
            st.session_state.resurfaced = brain.resurface()
            st.rerun()
        if b2.button("Dismiss"):
            st.session_state.resurface_dismissed = True
            st.rerun()


capture_tab, search_tab, browse_tab, graph_tab = st.tabs(["✍️ Capture", "🔎 Search", "🗂️ Browse", "🕸️ Graph"])


# --- capture -----------------------------------------------------------------

with capture_tab:
    with st.form("capture", clear_on_submit=True):
        text = st.text_area("What's on your mind?", height=160,
                            placeholder="A thought, a link, something you learned... (Markdown supported)")
        c1, c2 = st.columns([1, 3])
        note_type = c1.selectbox("Type", ["auto"] + TYPES)
        manual_tags = c2.text_input("Tags (optional, comma-separated)")
        submitted = st.form_submit_button("Save Note", type="primary")

    if submitted:
        if not text.strip():
            st.warning("Write something first.")
        else:
            with st.spinner("Saving, embedding and connecting..."):
                note, related = brain.save_note(
                    text,
                    type=None if note_type == "auto" else note_type,
                    tags=manual_tags,
                    auto_tag=auto_tag,
                )
            st.session_state.last_saved = (note, related, brain.last_tag_error)
            st.rerun()  # refresh sidebar counts

    if "last_saved" in st.session_state:
        note, related, tag_error = st.session_state.last_saved
        tag_txt = f" with tags {', '.join(f'`{t}`' for t in note.tags)}" if note.tags else ""
        st.success(f"Saved note #{note.id}{tag_txt}.")
        if tag_error:
            st.warning(f"Auto-tagging failed, so the note was saved without LLM tags. {tag_error}")
        if related:
            st.markdown("**This connects to:**")
            for other, score in related:
                st.markdown(f"- **{score:.0%}** · {other.title} <small>({friendly_time(other)})</small>",
                            unsafe_allow_html=True)

    todays = [n for n in brain.all_notes() if n.created.date() == datetime.now().date()]
    if todays:
        st.subheader(f"Today ({len(todays)})")
        for n in todays:
            render_note(n, key="today")


# --- search / "what are you working on?" -------------------------------------

with search_tab:
    query = st.text_area("Search your notes, or describe what you're working on",
                         height=100,
                         placeholder="e.g. 'planning a talk about vector databases' or 'that article on sleep'")
    c1, c2, c3 = st.columns(3)
    k = c1.slider("Results", 1, 30, 8)
    min_score = c2.slider("Minimum match", 0.0, 1.0, 0.2, 0.05)
    tag_filter = c3.multiselect("Only with tags", list(brain.tag_counts()))

    if query.strip():
        # Over-fetch when filtering by tag, so the filter doesn't starve the result list.
        results = brain.search(query, k=k * 4 if tag_filter else k, min_score=min_score)
        if tag_filter:
            results = [(n, s) for n, s in results if set(tag_filter) & set(n.tags)][:k]
        if results:
            for n, s in results:
                render_note(n, score=s, key="search")
        else:
            st.info("No notes match closely enough. Try lowering the minimum match.")


# --- browse ------------------------------------------------------------------

with browse_tab:
    notes = brain.all_notes()
    tag_counts = brain.tag_counts()
    c1, c2, c3, c4 = st.columns([3, 2, 3, 1])
    tags_sel = c1.multiselect("Tags", list(tag_counts), format_func=lambda t: f"{t} ({tag_counts[t]})")
    types_sel = c2.multiselect("Type", sorted({n.type for n in notes}))
    contains = c3.text_input("Contains (exact text)")
    oldest_first = c4.toggle("Oldest first")

    if tags_sel:
        notes = [n for n in notes if set(tags_sel) & set(n.tags)]
    if types_sel:
        notes = [n for n in notes if n.type in types_sel]
    if contains:
        notes = [n for n in notes if contains.lower() in n.text.lower()]
    if oldest_first:
        notes = notes[::-1]

    st.caption(f"{len(notes)} note{'s' if len(notes) != 1 else ''}")
    for n in notes[:200]:
        render_note(n, key="browse")
    if len(notes) > 200:
        st.caption("Showing the first 200. Narrow the filters to see more.")


# --- graph -------------------------------------------------------------------

with graph_tab:
    c1, c2, c3 = st.columns(3)
    edge_min = c1.slider("Minimum connection strength", 0.0, 1.0, MIN_CONNECTION_SCORE, 0.05)
    hide_isolated = c2.toggle("Hide unconnected notes", value=True)
    color_by = c3.radio("Color by", ["tag", "type"], horizontal=True)

    g = brain.graph(min_score=edge_min)
    if hide_isolated:
        g.remove_nodes_from([n for n, d in dict(g.degree()).items() if d == 0])

    if g.number_of_nodes() == 0:
        st.info("No connections at this strength yet. Add more notes or lower the slider.")
    else:
        groups: dict[str, str] = {}

        def color_for(note: Note) -> str:
            key = (note.tags[0] if note.tags else "untagged") if color_by == "tag" else note.type
            if key not in groups:
                groups[key] = PALETTE[len(groups) % len(PALETTE)]
            return groups[key]

        net = Network(height="680px", width="100%", bgcolor="#ffffff", font_color="#222222", cdn_resources="remote")
        for node_id, data in g.nodes(data=True):
            note: Note = data["note"]
            hover = f"#{note.id} · {note.created:%d %b %Y} · {', '.join(note.tags) or 'no tags'}\n\n{note.text[:400]}"
            net.add_node(node_id, label=strip_markup(note.title[:30]), title=strip_markup(hover), color=color_for(note),
                         size=10 + 4 * g.degree(node_id))
        for a, b, data in g.edges(data=True):
            w = data["weight"]
            net.add_edge(a, b, value=w, title=f"{w:.0%} similar", color={"color": "#bbbbbb", "opacity": 0.4 + w / 2})
        net.barnes_hut(gravity=-6000, spring_length=140)

        st.caption(f"{g.number_of_nodes()} notes · {g.number_of_edges()} connections. "
                   "Hover a node to read it; drag to rearrange.")
        st.iframe(net.generate_html(), height=700)
        st.markdown("  ".join(f"<span style='color:{c}'>●</span> {k}" for k, c in groups.items()),
                    unsafe_allow_html=True)
