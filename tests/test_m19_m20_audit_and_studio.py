"""
Unit tests for Milestone M19 (Cryptographic Provenance Figure Guard & LEAN Studio Cockpit)
and Milestone M20 (AST Integrity Scanner & God-Tier Audit Runner A1-A15).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_30b_publication_figures import (
    CryptographicProvenanceError,
    load_verified_receipt,
)
from src.lean.audit.ast_integrity_scanner import scan_repository_ast
from src.lean.studio.server import verify_crypto_ledger_chain


def test_m19_cryptographic_provenance_guard(tmp_path: Path) -> None:
    fake_json = tmp_path / "unverified_results.json"
    fake_json.write_text(json.dumps({"acc": 0.99}), encoding="utf-8")
    fake_ledger = tmp_path / "CRYPTO_LEDGER.jsonl"
    fake_ledger.write_text(
        json.dumps({"output_manifest": {"other.json": "0" * 64}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CryptographicProvenanceError):
        load_verified_receipt(fake_json, fake_ledger)


def test_m19_studio_ledger_verifier() -> None:
    res = verify_crypto_ledger_chain()
    assert res["valid"] is True
    assert res["entries"] >= 4


def test_m20_ast_integrity_scanner_clean() -> None:
    report = scan_repository_ast(["src/lean", "experiments"])
    assert report["passed"] is True
    assert report["violation_count"] == 0
    assert report["syntax_error_count"] == 0
