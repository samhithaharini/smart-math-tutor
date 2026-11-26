"""
agent.py — LangChain AI Agent Combining Gemini (LLM) + RAG + MCP Tools
======================================================================

This file is the "brain" where all 5 concepts come together:
  1. LLM (Gemini via `langchain-google-genai`):
     Reads the user's question, reasons about what it needs, and generates answers.
  2. RAG (`search_formulas` from `rag_engine.py`):
     Lets the agent look up custom formulas & constants from `formulas.txt`.
  3. MCP (`MultiServerMCPClient` connecting to `mcp_server.py`):
     Launches `mcp_server.py` over `stdio` and converts its MCP tools into
     LangChain-compatible tools automatically.
  4. LangChain (`langchain.agents.create_agent` & tool abstractions):
     Provides the unified glue so Gemini can invoke both Python RAG tools and
     external MCP server tools through one standard interface.
  5. AI Agent (ReAct-style Reason + Act loop inside `create_agent`):
     Instead of a single prompt->response call, the agent loops:
       Think -> Choose a Tool (RAG or MCP) -> Observe Tool Result -> Repeat -> Answer.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from rag_engine import search_formulas

# Load environment variables (such as GOOGLE_API_KEY) from `.env`.
load_dotenv()

MCP_SERVER_PATH = Path(__file__).parent / "mcp_server.py"

DEFAULT_MODEL = "gemini-4-hydrogen"
# If the primary model hits a temporary 503 (high demand) or 429 (quota limit),
# we gracefully fall back through other available Gemini models on your API key.
FALLBACK_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
]

SYSTEM_PROMPT = """You are "Smart Math & Formula Tutor", a helpful and accurate tutor.

You have access to two kinds of tools:
1. RAG Knowledge Base (`search_formulas`): Searches `formulas.txt` for formulas and custom constants (like Vishal's Constant, Kinetic Energy, Ohm's Law, Compound Interest, and the Quadratic Formula).
2. MCP Calculator Tools (`add`, `multiply`, `evaluate_expression`): Perform exact math calculations.

Rules you MUST follow:
- Whenever the user asks about a formula, definition, or custom constant, ALWAYS call `search_formulas` first—do NOT guess from memory.
- Whenever ANY arithmetic calculation is required, ALWAYS use the MCP calculator tools (`add`, `multiply`, or `evaluate_expression`)—never compute numbers in your head.
- If a question needs both a formula lookup AND math, call `search_formulas` first, then plug the values into the MCP calculator tool(s).
- Explain the final answer clearly step by step for a beginner.
"""


def _extract_text(content: Any) -> str:
    """Normalize LangChain message content (str or list of content blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
        return "\n".join(parts).strip()
    return str(content)


def _ensure_api_key() -> None:
    """Verify a Gemini API key is available in the environment."""
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise ValueError(
            "Missing Gemini API key. Copy `.env.example` to `.env` and set "
            "GOOGLE_API_KEY=your_actual_key."
        )


def _extract_thought_process(new_messages: list[Any], mcp_tool_names: set[str]) -> list[dict[str, Any]]:
    """
    Parse the intermediate AIMessage tool_calls and ToolMessage outputs from this turn
    so the UI can display WHICH layer (RAG vs. MCP) the agent chose and WHY.
    """
    steps_by_id: dict[str, dict[str, Any]] = {}
    ordered_steps: list[dict[str, Any]] = []

    for msg in new_messages:
        # 1. When the LLM decides to call a tool, it emits an AIMessage with `.tool_calls`.
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                tool_name = call.get("name", "unknown")
                call_id = call.get("id") or f"{tool_name}_{len(ordered_steps)}"
                source = "MCP (mcp_server.py)" if tool_name in mcp_tool_names else "RAG (rag_engine.py)"
                step = {
                    "source": source,
                    "tool": tool_name,
                    "args": call.get("args", {}),
                    "output": None,
                }
                steps_by_id[call_id] = step
                ordered_steps.append(step)

        # 2. When the tool finishes running, LangChain appends a ToolMessage with the result.
        elif getattr(msg, "type", None) == "tool":
            call_id = getattr(msg, "tool_call_id", None)
            output_text = _extract_text(getattr(msg, "content", ""))
            if call_id and call_id in steps_by_id:
                steps_by_id[call_id]["output"] = output_text
            else:
                tool_name = getattr(msg, "name", "unknown")
                source = "MCP (mcp_server.py)" if tool_name in mcp_tool_names else "RAG (rag_engine.py)"
                ordered_steps.append(
                    {
                        "source": source,
                        "tool": tool_name,
                        "args": {},
                        "output": output_text,
                    }
                )

    return ordered_steps


async def ask_tutor(query: str, chat_history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """
    Run one turn of the Smart Math & Formula Tutor agent asynchronously.

    Returns:
        {
            "answer": str,          # Final response from Gemini
            "steps": list[dict],    # Ordered tool calls (RAG vs. MCP) + outputs
        }
    """
    _ensure_api_key()

    # --- CONCEPT: MCP Client ---
    # MultiServerMCPClient spawns `mcp_server.py` as a subprocess using the current
    # Python interpreter (`sys.executable`) and talks over stdin/stdout (`stdio`).
    mcp_client = MultiServerMCPClient(
        {
            "math_calculator": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(MCP_SERVER_PATH)],
            }
        }
    )

    # Fetch MCP tools and convert them into LangChain tools.
    mcp_tools = await mcp_client.get_tools()
    mcp_tool_names = {tool.name for tool in mcp_tools}

    # --- CONCEPT: Combining MCP + RAG Tools ---
    # To the Agent, MCP calculator tools and our local Python RAG tool look identical!
    all_tools = [*mcp_tools, search_formulas]

    # Include prior chat turns (role + content only) so follow-up questions work naturally.
    input_messages: list[dict[str, str]] = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in (chat_history or [])
        if msg.get("role") in ("user", "assistant")
    ]
    input_messages.append({"role": "user", "content": query})

    # Build ordered list of candidate models (user's GEMINI_MODEL or DEFAULT_MODEL first).
    primary_model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    candidate_models = list(dict.fromkeys([primary_model, *FALLBACK_MODELS]))

    result = None
    used_model = primary_model
    last_error: Exception | None = None

    for model_name in candidate_models:
        # --- CONCEPT: LLM (Gemini) ---
        # `max_retries=0` fails fast on 503/429 so our fallback loop switches models in seconds
        # instead of stalling on long exponential-backoff retries.
        llm = ChatGoogleGenerativeAI(model=model_name, max_retries=0)

        # --- CONCEPT: AI Agent (`create_agent`) ---
        # `langchain.agents.create_agent` builds the agent loop that lets Gemini
        # decide which tool(s) to call, read their outputs, and formulate a final reply.
        agent = create_agent(
            model=llm,
            tools=all_tools,
            system_prompt=SYSTEM_PROMPT,
        )
        try:
            result = await agent.ainvoke({"messages": input_messages})
            used_model = model_name
            break
        except Exception as exc:
            err_msg = str(exc)
            if "503" in err_msg or "429" in err_msg or "UNAVAILABLE" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                last_error = exc
                continue
            raise

    if result is None:
        raise RuntimeError(
            f"All Gemini models are currently unavailable or rate-limited. Last error: {last_error}"
        ) from last_error

    all_messages = result.get("messages", [])

    # Only inspect messages generated during THIS turn for the Thought Process box.
    new_messages = all_messages[len(input_messages):]
    thought_steps = _extract_thought_process(new_messages, mcp_tool_names)

    # The last message in `all_messages` is the agent's final answer to the user.
    final_answer = _extract_text(all_messages[-1].content) if all_messages else "No response generated."

    return {
        "answer": final_answer,
        "steps": thought_steps,
        "model": used_model,
    }


if __name__ == "__main__":
    # Quick terminal smoke test: `python agent.py`
    sample_query = "What is Vishal's Constant multiplied by 10?"
    print(f"Query: {sample_query}\n")
    response = asyncio.run(ask_tutor(sample_query))
    print("Thought Process:")
    for i, step in enumerate(response["steps"], start=1):
        print(f"  {i}. [{step['source']}] {step['tool']}({step['args']}) -> {step['output']}")
    print(f"\nFinal Answer:\n{response['answer']}")
