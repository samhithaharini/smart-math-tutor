"""
mcp_server.py — Standalone MCP (Model Context Protocol) Tool Server
==================================================================

CONCEPT — MCP (Model Context Protocol):
MCP is an open standard that lets an AI application connect to external tools
and data sources in a universal, framework-independent way (like USB-C for AI).

Notice what is NOT imported in this file:
- NO LangChain
- NO Gemini / LLM SDK
- NO Streamlit

This server only speaks the MCP protocol over `stdio` (standard input/output).
Any MCP-compatible client (LangChain, Claude Desktop, Cursor, etc.) can launch
this script as a subprocess and call its tools without knowing how they are
implemented internally.
"""

import ast
import operator
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server with a human-readable name.
mcp = FastMCP("SmartMath Calculator Server")

# Safe operators allowed by our expression evaluator (never use raw eval()!).
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval_node(node: ast.AST) -> float:
    """Recursively evaluate a parsed Python math AST node safely."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return float(_BINARY_OPERATORS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return float(_UNARY_OPERATORS[type(node.op)](_safe_eval_node(node.operand)))
    raise ValueError(f"Unsupported math operation in expression: {ast.dump(node)}")


# `@mcp.tool()` registers a plain Python function as an MCP tool.
# FastMCP reads the type hints and docstring to tell the AI client what the tool does.
@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together (a + b) and return the exact sum."""
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together (a * b) and return the exact product."""
    return a * b


@mcp.tool()
def evaluate_expression(expression: str) -> float:
    """
    Safely evaluate an arithmetic math expression string and return the result.
    Supports +, -, *, /, //, %, ** (exponentiation), and parentheses.
    Example: "(42 * 10) + 5**2" -> 445.0
    """
    # Allow caret `^` as exponentiation too, since formulas often use `^` instead of `**`.
    normalized = expression.strip().replace("^", "**")
    tree = ast.parse(normalized, mode="eval")
    return _safe_eval_node(tree)


if __name__ == "__main__":
    # IMPORTANT: Because `transport="stdio"` uses stdout for JSON-RPC messages,
    # never use `print(...)` inside a stdio MCP server—it would corrupt the protocol!
    mcp.run(transport="stdio")
