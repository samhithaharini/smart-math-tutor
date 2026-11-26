# Smart Math & Formula Tutor

A minimal, beginner-friendly Python project that demonstrates **LLM**, **RAG**, **MCP**, **LangChain**, and **AI Agents** working together in one clean application.

---

## How the 5 Concepts Fit Together

| Concept | File | Role in This Project |
|---|---|---|
| **Knowledge Base** | `formulas.txt` | Plain text file containing 5 formulas (including the fictional `Vishal's Constant = 42` to prove RAG works). |
| **MCP** | `mcp_server.py` | Standalone `FastMCP` server over `stdio` exposing `add`, `multiply`, and `evaluate_expression` — **zero LangChain imports**. |
| **RAG** | `rag_engine.py` | Embeds `formulas.txt` locally with `sentence-transformers/all-MiniLM-L6-v2` into an in-memory `Chroma` vector store and exposes `search_formulas` as a LangChain `@tool`. |
| **LLM + LangChain + Agent** | `agent.py` | Connects to `mcp_server.py` via `MultiServerMCPClient`, combines MCP tools + the RAG tool, and runs an async `create_agent` loop powered by Google Gemini (`ChatGoogleGenerativeAI`). |
| **UI** | `app.py` | Streamlit chat interface showing the final answer and an expandable **Agent Thought Process** box revealing whether the agent chose **RAG**, **MCP**, or **both**. |

---

## Step-by-Step Setup

### 1. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On the very first run, `sentence-transformers/all-MiniLM-L6-v2` (~90 MB) will be downloaded once and cached locally by HuggingFace.

### 3. Configure Your Gemini API Key

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Copy `.env.example` to `.env`:
   - Windows PowerShell: `Copy-Item .env.example .env`
   - macOS / Linux: `cp .env.example .env`
3. Open `.env` and paste your key:
   ```env
   GOOGLE_API_KEY=your_actual_gemini_api_key
   # Optional: override the default model (gemini-4-hydrogen):
   # GEMINI_MODEL=gemini-3.8-flash
   ```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

*(Optional)* You can also test the agent directly in the terminal before opening Streamlit:
```bash
python agent.py
```

---

## 3 Test Prompts to Try

Open the **Agent Thought Process (Tools Used)** expander under each answer to verify which layer fired!

### 1. Triggers ONLY the MCP Calculator
> **Prompt:** `What is (125 * 48) + 360?`
- **Expected Thought Process:** Calls `MCP (mcp_server.py)` (`evaluate_expression`, or `multiply` + `add`). Does **not** call `search_formulas`.

### 2. Triggers ONLY RAG
> **Prompt:** `What is Vishal's Constant and why does it exist?`
- **Expected Thought Process:** Calls `RAG (rag_engine.py)` -> `search_formulas` and reads `Vishal's Constant (V_c) = 42` from `formulas.txt`. Does **not** call any MCP calculator tool.

### 3. Triggers BOTH RAG and MCP in One Query
> **Prompt:** `Look up Vishal's Constant, multiply it by 15, and then add 8.`
- **Expected Thought Process:**
  1. Calls `RAG (rag_engine.py)` -> `search_formulas` to discover `Vishal's Constant = 42`.
  2. Calls `MCP (mcp_server.py)` -> `multiply(42, 15)` -> `630`, then `add(630, 8)` -> `638` (or `evaluate_expression("42 * 15 + 8")`).
