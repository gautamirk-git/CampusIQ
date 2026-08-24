"""Streamlit chat frontend for CampusIQ."""

import base64
import os
from pathlib import Path

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# TODO: replace with a real contact address before sharing this publicly.
CONTACT_EMAIL = "your-email@example.com"

# Optional: if set, visitors must enter this code before the chat unlocks.
# Leave unset for open local dev. Set via `$env:CAMPUSIQ_ACCESS_CODE = "..."`
# (PowerShell) or your hosting platform's env/secrets settings.
ACCESS_CODE = os.environ.get("CAMPUSIQ_ACCESS_CODE", "")

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
BACKGROUND_IMAGE = ASSETS_DIR / "tech_tower.jpg"

NAVY = "#003057"
GOLD = "#B3A369"

st.set_page_config(page_title="CampusIQ", page_icon="🎓", layout="centered")


@st.cache_data
def _image_as_base64(path: Path) -> str | None:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _inject_styles() -> None:
    bg_b64 = _image_as_base64(BACKGROUND_IMAGE)
    background_layer = (
        f"linear-gradient(rgba(0, 48, 87, 0.88), rgba(0, 48, 87, 0.93)), "
        f"url('data:image/jpeg;base64,{bg_b64}')"
        if bg_b64
        else f"linear-gradient(160deg, {NAVY} 0%, #001a30 100%)"
    )

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: {background_layer};
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .campusiq-header {{
            text-align: center;
            padding: 1.75rem 1rem 1rem 1rem;
        }}
        .campusiq-header h1 {{
            color: #ffffff;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.1rem;
            letter-spacing: 0.02em;
        }}
        .campusiq-header .tagline {{
            color: {GOLD};
            font-size: 1.05rem;
            font-weight: 500;
        }}
        .campusiq-badge {{
            display: inline-block;
            margin-top: 0.6rem;
            padding: 0.25rem 0.85rem;
            border: 1px solid rgba(255, 255, 255, 0.35);
            border-radius: 999px;
            color: #e8e8e8;
            font-size: 0.78rem;
            background: rgba(0, 0, 0, 0.15);
        }}

        [data-testid="stChatMessage"] {{
            background: rgba(255, 255, 255, 0.96);
            border-radius: 0.75rem;
            padding: 0.5rem 0.75rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
        }}
        [data-testid="stChatMessage"] * {{
            color: #1a1a1a !important;
        }}

        [data-testid="stChatInput"] {{
            border-radius: 0.75rem;
        }}

        .campusiq-footer {{
            text-align: center;
            color: #dfe6ed;
            font-size: 0.8rem;
            padding: 2rem 1rem 1rem 1rem;
            line-height: 1.6;
        }}
        .campusiq-footer a {{
            color: {GOLD};
            text-decoration: none;
        }}

        .campusiq-subtitle {{
            text-align: center;
            color: #dfe6ed;
            font-size: 0.95rem;
            padding: 0 1rem 1rem 1rem;
        }}

        .stApp label {{
            color: #dfe6ed !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_styles()

st.markdown(
    """
    <div class="campusiq-header">
        <h1>🎓 CampusIQ</h1>
        <div class="tagline">Your guide to admissions, cost, majors, and campus life</div>
        <div class="campusiq-badge">Independent project — not affiliated with or endorsed by Georgia Tech</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="campusiq-subtitle">Ask about admissions, tuition, housing, or '
    "majors. Answers are grounded in official Georgia Tech pages the assistant "
    'has read — it will say "I don\'t know" if something isn\'t covered.</div>',
    unsafe_allow_html=True,
)

if ACCESS_CODE and not st.session_state.get("authenticated"):
    with st.form("access_gate"):
        entered_code = st.text_input("Access code", type="password")
        submitted = st.form_submit_button("Enter")
    if submitted:
        if entered_code == ACCESS_CODE:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect access code.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']}** (distance: {s['distance']:.3f})")
                    st.text(s["excerpt"] + "...")

question = st.chat_input("Ask a question about Georgia Tech...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/ask", json={"question": question}, timeout=60
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["answer"]
                sources = data.get("sources", [])
            except requests.exceptions.ConnectionError:
                answer = (
                    f"⚠️ Couldn't reach the backend at {BACKEND_URL} — is it running?"
                )
                sources = []
            except requests.exceptions.HTTPError as exc:
                # Backend responded but with an error status — show its actual
                # detail message (e.g. "credit balance too low") instead of the
                # generic "500 Server Error" text.
                try:
                    detail = exc.response.json().get("detail", exc.response.text)
                except ValueError:
                    detail = exc.response.text
                answer = f"⚠️ {detail}"
                sources = []
            except requests.exceptions.RequestException as exc:
                answer = f"⚠️ Request to the backend failed: {exc}"
                sources = []

        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"**{s['source']}** (distance: {s['distance']:.3f})")
                    st.text(s["excerpt"] + "...")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

st.markdown(
    f"""
    <div class="campusiq-footer">
        CampusIQ is an independent student project and is not affiliated with,
        endorsed by, or sponsored by the Georgia Institute of Technology.<br>
        Questions or feedback? <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a><br>
        Background photo:
        <a href="https://commons.wikimedia.org/wiki/File:Tech_Tower_and_skyline.jpg" target="_blank">
        "Tech Tower and skyline"</a> by TedMiles, licensed under
        <a href="https://creativecommons.org/licenses/by-sa/3.0/" target="_blank">CC BY-SA 3.0</a>.
    </div>
    """,
    unsafe_allow_html=True,
)
