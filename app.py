"""
app.py — Streamlit Chat UI for Smart Math & Formula Tutor
=========================================================

CONCEPT — Presentation Layer:
Streamlit wraps our async AI Agent (`agent.py`) in a beginner-friendly web chat
interface. Crucially, it separates:
  - What the user sees as the Final Answer
  - What happened behind the scenes in the "Agent Thought Process" expander
    (showing whether the agent picked RAG, MCP, or both).
"""

import asyncio
import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from agent import ask_tutor

load_dotenv()

st.set_page_config(
    page_title="Smart Math & Formula Tutor",
    page_icon="🧮",
    layout="centered",
)

# Sample prompts demonstrating each routing path of the Agent.
SAMPLE_PROMPTS = {
    "MCP Only (Calculator)": "What is (125 * 48) + 360?",
    "RAG Only (Formula Lookup)": "What is Vishal's Constant and why does it exist?",
    "Both RAG + MCP": "Look up Vishal's Constant, multiply it by 15, and then add 8.",
}


def _render_thought_process(
    steps: list[dict[str, Any]], key_prefix: str, model_name: str | None = None
) -> None:
    """Render an expandable trace showing which tool (RAG vs. MCP) the agent called."""
    with st.expander("Agent Thought Process (Tools Used)", expanded=False):
        if model_name:
            st.caption(f"LLM Model used: `{model_name}`")
        if not steps:
            st.info("No external tools were needed — answered directly by the LLM (Gemini).")
            return

        for idx, step in enumerate(steps, start=1):
            source = step.get("source", "Unknown")
            badge = "📚 RAG" if source.startswith("RAG") else "🔧 MCP"
            st.markdown(f"**Step {idx} — {badge} (`{source}`)**")
            st.markdown(f"- **Tool:** `{step.get('tool')}`")
            st.markdown(f"- **Arguments:** `{step.get('args')}`")
            st.text_area(
                label=f"Tool Output {key_prefix}-{idx}",
                value=str(step.get("output", "")),
                height=80,
                disabled=True,
                label_visibility="collapsed",
                key=f"{key_prefix}_{idx}",
            )
            if idx < len(steps):
                st.divider()


# --- Sidebar: Learning Guide & Quick Test Buttons ---
with st.sidebar:
    st.header("How This App Works")
    st.markdown(
        """
        - **LLM (`Gemini`)**: Reasons about your question and writes the explanation.
        - **RAG (`rag_engine.py`)**: Searches `formulas.txt` via local embeddings + Chroma.
        - **MCP (`mcp_server.py`)**: Runs standalone math tools (`add`, `multiply`, `evaluate_expression`) over `stdio`.
        - **LangChain + Agent (`agent.py`)**: Connects Gemini, RAG, and MCP into one reasoning loop.
        """
    )

    has_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    if has_key:
        st.success("Gemini API key detected in environment.")
    else:
        st.warning("No `GOOGLE_API_KEY` found. Copy `.env.example` to `.env` and add your key.")

    st.subheader("Try a Test Prompt")
    for label, prompt_text in SAMPLE_PROMPTS.items():
        if st.button(label, use_container_width=True):
            st.session_state.queued_prompt = prompt_text

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- Main Chat Interface ---
st.title("🧮 Smart Math & Formula Tutor")
st.caption("A minimal educational demo of **LLM + RAG + MCP + LangChain + AI Agents**")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render prior conversation messages.
for msg_index, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "steps" in msg:
            _render_thought_process(
                msg["steps"],
                key_prefix=f"hist_{msg_index}",
                model_name=msg.get("model"),
            )

# Accept input either from the chat box or from a clicked sidebar sample button.
queued_prompt = st.session_state.pop("queued_prompt", None)
user_prompt = st.chat_input("Ask about a formula or a math problem...") or queued_prompt

if user_prompt:
    # Show the user's message immediately.
    with st.chat_message("user"):
        st.markdown(user_prompt)

    history_before_turn = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Run the async agent and display its reply + thought process.
    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking and selecting tools (RAG / MCP)..."):
            try:
                result = asyncio.run(ask_tutor(user_prompt, chat_history=history_before_turn))
                answer = result["answer"]
                steps = result["steps"]
                used_model = result.get("model")
            except Exception as exc:
                answer = f"⚠️ Error running agent: `{exc}`"
                steps = []
                used_model = None

        st.markdown(answer)
        _render_thought_process(
            steps,
            key_prefix=f"turn_{len(st.session_state.messages)}",
            model_name=used_model,
        )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "steps": steps, "model": used_model}
    )
