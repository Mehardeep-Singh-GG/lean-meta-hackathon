# File: tests/test_m13_colab_crypto.py
"""Automated Gate Verification Suite for Milestone M13 (G1 through G6)."""

import json
import os
from pathlib import Path
import pytest

from src.lean.audit.crypto_ledger import (
    CryptoLedgerTamperError,
    append_signed_receipt,
    verify_ledger_chain,
)
from src.lean.audit.failure_ledger import check_blocking_failures
from src.lean.cloud.colab_bridge import ColabBridgeClient, ColabBridgeError


def test_m13_crypto_ledger_chain_and_tamper_detection(tmp_path: Path) -> None:
    """Verify Gate M13-G5: Merkle hash-chain integrity and single-bit tamper detection."""
    ledger_file = tmp_path / "CRYPTO_LEDGER.jsonl"
    in_file = tmp_path / "input.json"
    out_file = tmp_path / "output.json"
    in_file.write_text('{"seed": 42}', encoding="utf-8")
    out_file.write_text('{"acc": 0.88}', encoding="utf-8")

    r0 = append_signed_receipt(
        milestone="M13",
        job_id="test_job_0",
        execution_mode="local_unit_test",
        script_path=Path(__file__),
        config={"lr": 1e-4},
        input_files=[in_file],
        output_files=[out_file],
        ledger_path=ledger_file,
    )
    r1 = append_signed_receipt(
        milestone="M13",
        job_id="test_job_1",
        execution_mode="local_unit_test",
        script_path=Path(__file__),
        config={"lr": 2e-4},
        input_files=[in_file],
        output_files=[out_file],
        ledger_path=ledger_file,
    )
    assert r1["prev_receipt_sha256"] == r0["receipt_sha256"]
    ok, count, head = verify_ledger_chain(ledger_file)
    assert ok and count == 2 and head == r1["receipt_sha256"]

    # Tamper with config in row 0 and verify CryptoLedgerTamperError is raised
    lines = ledger_file.read_text(encoding="utf-8").splitlines()
    row0 = json.loads(lines[0])
    row0["config"]["lr"] = 999.0
    lines[0] = json.dumps(row0)
    ledger_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(CryptoLedgerTamperError):
        verify_ledger_chain(ledger_file)


def test_m13_blocking_failures() -> None:
    """Verify Gate M13-G6: Zero blocking failures for milestone M13."""
    assert not check_blocking_failures("M13")
