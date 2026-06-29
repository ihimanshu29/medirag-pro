"""
MediRAG Pro — Streamlit Frontend.

Talks to the FastAPI backend via HTTP (not importing pipeline directly).
Backend URL is discovered from environment variable API_BACKEND_URL.

Deployment:
  Local:       API_BACKEND_URL=http://localhost:8000  (default)
  Free cloud:  API_BACKEND_URL=https://your-hf-space.hf.space
  VPS:         API_BACKEND_URL=https://api.yourdomain.com
               or API_BACKEND_URL=http://YOUR_IP:8000
"""
import os
import uuid

import requests
import streamlit as st

# ── Backend URL — driven by environment, never hardcoded ─────────────────────
_API_BACKEND_URL = os.environ.get("API_BACKEND_URL", "http://localhost:8000").rstrip("/")
API_BASE = f"{_API_BACKEND_URL}/api/v1"
HEALTH_URL = f"{_API_BACKEND_URL}/health"

st.set_page_config(
    page_title="MediRAG Pro",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "cache_hits" not in st.session_state:
    st.session_state.cache_hits = 0


# ── Helper functions ──────────────────────────────────────────────────────────
def check_health() -> dict:
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        return r.json()
    except Exception:
        return {"status": "down", "components": {}}


def send_query(query: str, source_filter: str | None = None) -> dict:
    payload = {
        "query": query,
        "session_id": st.session_state.session_id,
        "source_filter": source_filter or None,
    }
    r = requests.post(f"{API_BASE}/chat", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def upload_document(file) -> dict:
    r = requests.post(
        f"{API_BASE}/ingest",
        files={"file": (file.name, file.getvalue(), "application/pdf")},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def submit_feedback(query: str, answer: str, rating: int) -> None:
    try:
        requests.post(
            f"{API_BASE}/feedback",
            json={
                "session_id": st.session_state.session_id,
                "query": query,
                "answer": answer,
                "rating": rating,
            },
            timeout=5,
        )
    except Exception:
        pass


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🩺 MediRAG Pro")
    st.caption("Medical Knowledge Assistant")

    # Health status
    health = check_health()
    status = health.get("status", "down")
    status_color = {"healthy": "🟢", "degraded": "🟡", "down": "🔴"}.get(status, "🔴")
    st.markdown(f"**System:** {status_color} {status.upper()}")

    if health.get("components"):
        comps = health["components"]
        qdrant_status = comps.get("qdrant", {}).get("status", "unknown")
        pg_status = comps.get("postgres", {}).get("status", "unknown")
        env = comps.get("environment", "unknown")
        st.caption(f"Qdrant: {qdrant_status} | DB: {pg_status} | Env: {env}")

    st.divider()

    # Session info
    st.markdown("**Session**")
    st.code(st.session_state.session_id[:8] + "...", language=None)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.total_queries)
    with col2:
        hit_rate = (
            f"{st.session_state.cache_hits / st.session_state.total_queries:.0%}"
            if st.session_state.total_queries > 0 else "—"
        )
        st.metric("Cache Hit", hit_rate)

    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.cache_hits = 0
        st.rerun()

    st.divider()

    # Document upload
    st.markdown("**📄 Ingest Document**")
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Upload a medical PDF to add to the knowledge base",
    )
    if uploaded_file and st.button("Ingest Document", use_container_width=True):
        with st.spinner(f"Processing {uploaded_file.name}..."):
            try:
                result = upload_document(uploaded_file)
                st.success(
                    f"✅ {result['chunks_created']} chunks indexed\n"
                    f"📊 {result['tables_extracted']} tables extracted"
                )
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.divider()

    # Source filter
    st.markdown("**🔍 Filter by Source**")
    source_filter = st.text_input(
        "Filename (optional)",
        placeholder="e.g. medical_textbook.pdf",
        help="Restrict retrieval to a specific document",
    )

    st.divider()
    st.caption(
        "⚠️ Not a substitute for professional medical advice. "
        "Always consult a qualified healthcare provider."
    )


# ── Main chat area ────────────────────────────────────────────────────────────
st.title("MediRAG Pro 🩺")
st.caption("Ask medical questions grounded in your knowledge base documents.")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(
                f"📚 Sources ({len(msg['sources'])}) — "
                f"Confidence: {msg.get('confidence', 0):.0%}",
                expanded=False,
            ):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**[{i}] {src['source_file']}** "
                        f"{'— p.' + str(src['page']) if src.get('page') else ''} "
                        f"{'— ' + src['section'] if src.get('section') else ''} "
                        f"*(score: {src['score']:.3f})*"
                    )
                    st.markdown(f"> {src['content'][:300]}{'...' if len(src['content']) > 300 else ''}")
                    if i < len(msg["sources"]):
                        st.divider()

        if msg["role"] == "assistant":
            badges = []
            if msg.get("cache_hit"):
                badges.append("⚡ Cached")
            if msg.get("is_emergency"):
                badges.append("🚨 Emergency")
            if msg.get("latency_ms"):
                badges.append(f"⏱ {msg['latency_ms']:.0f}ms")
            if badges:
                st.caption(" · ".join(badges))

            col1, col2, col3 = st.columns([1, 1, 8])
            msg_idx = st.session_state.messages.index(msg)
            with col1:
                if st.button("👍", key=f"up_{msg_idx}"):
                    submit_feedback(msg.get("query", ""), msg["content"], rating=1)
                    st.toast("Thanks for the feedback!")
            with col2:
                if st.button("👎", key=f"down_{msg_idx}"):
                    submit_feedback(msg.get("query", ""), msg["content"], rating=-1)
                    st.toast("Feedback noted — we'll improve!")


# ── Chat input ────────────────────────────────────────────────────────────────
if query := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer..."):
            try:
                response = send_query(
                    query,
                    source_filter=source_filter if source_filter else None,
                )

                answer = response["answer"]
                sources = response.get("sources", [])
                confidence = response.get("confidence", 0.0)
                is_emergency = response.get("is_emergency", False)
                cache_hit = response.get("cache_hit", False)
                latency_ms = response.get("latency_ms", 0)

                st.session_state.total_queries += 1
                if cache_hit:
                    st.session_state.cache_hits += 1

                if is_emergency:
                    st.error(answer)
                else:
                    st.markdown(answer)

                if sources:
                    with st.expander(
                        f"📚 Sources ({len(sources)}) — Confidence: {confidence:.0%}",
                        expanded=confidence < 0.5,
                    ):
                        for i, src in enumerate(sources, 1):
                            st.markdown(
                                f"**[{i}] {src['source_file']}** "
                                f"{'— p.' + str(src['page']) if src.get('page') else ''} "
                                f"{'— ' + src['section'] if src.get('section') else ''} "
                                f"*(score: {src['score']:.3f})*"
                            )
                            st.markdown(
                                f"> {src['content'][:300]}"
                                f"{'...' if len(src['content']) > 300 else ''}"
                            )
                            if i < len(sources):
                                st.divider()

                badges = []
                if cache_hit:
                    badges.append("⚡ Cached")
                if is_emergency:
                    badges.append("🚨 Emergency")
                badges.append(f"⏱ {latency_ms:.0f}ms")
                st.caption(" · ".join(badges))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "confidence": confidence,
                    "is_emergency": is_emergency,
                    "cache_hit": cache_hit,
                    "latency_ms": latency_ms,
                    "query": query,
                })

            except requests.exceptions.ConnectionError:
                st.error(
                    f"❌ Cannot connect to the API at `{_API_BACKEND_URL}`.\n\n"
                    "Set `API_BACKEND_URL` environment variable to the correct backend URL."
                )
            except Exception as e:
                st.error(f"❌ Error: {e}")
