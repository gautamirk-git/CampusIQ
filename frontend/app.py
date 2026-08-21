"""Streamlit chat frontend for CampusIQ."""

import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="CampusIQ", page_icon="🎓")
st.title("🎓 CampusIQ")
st.caption(
    "Ask about courses, admissions, cost, or general info. Answers are grounded "
    "in the ingested university document — the assistant will say \"I don't know\" "
    "if something isn't covered."
)

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

question = st.chat_input("Ask a question about the university...")

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
