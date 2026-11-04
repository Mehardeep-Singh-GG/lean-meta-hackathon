"""
Unit Tests for Milestone M17:
Long-Horizon Agentic Workflows & 5-Domain Non-Overfitting Stress Test.
"""

from __future__ import annotations

from pathlib import Path

from src.lean.agentic.react_loop_engine import (
    parse_react_action_or_final,
    verify_zero_mock_ast_integrity,
)
from src.lean.agentic.sandbox_tools import AgentToolSuite


def test_sandbox_tools_real_execution_and_sha256() -> None:
    suite = AgentToolSuite()

    # 1. PythonASTMathSandbox
    m_res = suite.dispatch("python_math_exec", "384 * 1.25 * 24 * 0.18")
    assert m_res["status"] == "ok"
    assert m_res["output"] == "2073.6"
    assert len(m_res["io_sha256"]) == 64

    # Security check: arbitrary code execution blocked
    m_bad = suite.dispatch("python_math_exec", "__import__('os').system('echo pwn')")
    assert m_bad["status"] == "error"

    # 2. BM25SearchTool
    b_res = suite.dispatch("bm25_knowledge_lookup", "Project Titan Zurich nodes")
    assert b_res["status"] == "ok"
    assert "384" in b_res["output"]

    # 3. InMemorySQLiteTool
    s_res = suite.dispatch(
        "sql_table_query",
        "SELECT SUM(hourly_cost_usd) AS total FROM server_fleet WHERE datacenter='zurich' AND status='active'",
    )
    assert s_res["status"] == "ok"
    assert "49.0" in s_res["output"]

    # Write-protection check
    s_bad = suite.dispatch("sql_table_query", "DROP TABLE server_fleet")
    assert s_bad["status"] == "error"

    # 4. PytestCodeVerifier
    c_good = suite.dispatch("code_linter_check", "def sq(x): return x*x\nassert sq(9) == 81")
    assert c_good["status"] == "ok"
    assert c_good["output"] == "ALL_TESTS_PASSED_OK"

    c_bad = suite.dispatch("code_linter_check", "def sq(x): return x+x\nassert sq(9) == 81")
    assert c_bad["status"] == "error"
    assert "PythonTestFailure" in c_bad["output"]


def test_parse_react_action_and_zero_mock_ast() -> None:
    act = parse_react_action_or_final('Thought: query nodes.\nAction: {"tool": "bm25_knowledge_lookup", "arg": "Project Titan"}')
    assert act["type"] == "action"
    assert act["syntax_valid"] is True
    assert act["tool"] == "bm25_knowledge_lookup"

    fin = parse_react_action_or_final("Thought: done.\nFinal Answer: 2073.6 CHF")
    assert fin["type"] == "final"
    assert fin["answer"] == "2073.6 CHF"

    err = parse_react_action_or_final("Thought: broken json {tool: 123")
    assert err["type"] == "syntax_error"
    assert err["syntax_valid"] is False

    root = Path(__file__).resolve().parents[1]
    ast_res = verify_zero_mock_ast_integrity(
        [
            root / "src" / "lean" / "agentic" / "sandbox_tools.py",
            root / "src" / "lean" / "agentic" / "react_loop_engine.py",
        ]
    )
    assert ast_res["zero_mock_passed"] is True
