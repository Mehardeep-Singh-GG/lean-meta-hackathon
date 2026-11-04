"""
Failure Ledger management and automated reporting for Project LEAN.

Maintains audit/FAILURES.jsonl and generates audit/FAILURES.md.
Enforces that no milestone closes with open S0 or S1 issues.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

AUDIT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "audit"
LEDGER_PATH = AUDIT_DIR / "FAILURES.jsonl"
SCHEMA_PATH = AUDIT_DIR / "failures_schema.json"
MD_PATH = AUDIT_DIR / "FAILURES.md"


def load_failures() -> List[Dict[str, Any]]:
    """Load all records from FAILURES.jsonl."""
    if not LEDGER_PATH.exists():
        return []
    records = []
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_failures(records: List[Dict[str, Any]]) -> None:
    """Save records to FAILURES.jsonl."""
    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    render_markdown_summary()


def validate_failure_record(record: Dict[str, Any]) -> None:
    """Validate record structure against expected keys and values."""
    required = [
        "id", "milestone", "opened", "category", "severity",
        "symptom", "detected_by", "root_cause", "five_whys",
        "fix", "verification", "regression_test", "status", "escalation_level"
    ]
    for k in required:
        if k not in record:
            raise ValueError(f"Missing required field in failure record: {k}")
    if len(record["five_whys"]) != 5:
        raise ValueError("five_whys must contain exactly 5 entries.")
    if record["severity"] not in ["S0", "S1", "S2", "S3"]:
        raise ValueError(f"Invalid severity: {record['severity']}")
    if record["status"] not in ["OPEN", "INVESTIGATING", "FIXING", "VERIFYING", "CLOSED"]:
        raise ValueError(f"Invalid status: {record['status']}")


def log_failure(
    failure_id: str,
    milestone: str,
    category: str,
    severity: str,
    symptom: str,
    detected_by: str,
    root_cause: str,
    five_whys: List[str],
    fix: str,
    verification: str,
    regression_test: str,
    status: str = "OPEN",
    escalation_level: int = 1,
    closed: Optional[str] = None
) -> Dict[str, Any]:
    """Add a new failure record to the ledger."""
    records = load_failures()
    existing_ids = {r["id"] for r in records}
    if failure_id in existing_ids:
        raise ValueError(f"Failure ID {failure_id} already exists.")
    
    now_iso = datetime.now(timezone.utc).isoformat()
    record = {
        "id": failure_id,
        "milestone": milestone,
        "opened": now_iso,
        "closed": closed,
        "category": category,
        "severity": severity,
        "symptom": symptom,
        "detected_by": detected_by,
        "root_cause": root_cause,
        "five_whys": five_whys,
        "fix": fix,
        "verification": verification,
        "regression_test": regression_test,
        "status": status,
        "escalation_level": escalation_level
    }
    validate_failure_record(record)
    records.append(record)
    save_failures(records)
    return record


def update_failure_status(failure_id: str, status: str, fix_notes: Optional[str] = None) -> Dict[str, Any]:
    """Update status of an existing failure record."""
    records = load_failures()
    target = None
    for r in records:
        if r["id"] == failure_id:
            target = r
            break
    if not target:
        raise KeyError(f"Failure ID {failure_id} not found.")
    
    target["status"] = status
    if fix_notes:
        target["fix"] = fix_notes
    if status == "CLOSED":
        target["closed"] = datetime.now(timezone.utc).isoformat()
    else:
        target["closed"] = None
        
    validate_failure_record(target)
    save_failures(records)
    return target


def check_blocking_failures(milestone: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return all open S0 or S1 failures (blocking milestone completion)."""
    records = load_failures()
    blocking = []
    for r in records:
        if milestone and r["milestone"] != milestone:
            continue
        if r["severity"] in ["S0", "S1"] and r["status"] != "CLOSED":
            blocking.append(r)
    return blocking


def render_markdown_summary() -> None:
    """Render audit/FAILURES.md tracking view."""
    records = load_failures()
    total = len(records)
    open_count = sum(1 for r in records if r["status"] != "CLOSED")
    s0_open = sum(1 for r in records if r["severity"] == "S0" and r["status"] != "CLOSED")
    s1_open = sum(1 for r in records if r["severity"] == "S1" and r["status"] != "CLOSED")
    
    md = [
        "# Project LEAN: Failure Ledger",
        "",
        f"**Last Updated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"**Total Tracked:** {total} | **Active/Open:** {open_count} | **Blocking (S0/S1):** {s0_open + s1_open}",
        "",
        "## Blocker Status",
        ""
    ]
    
    if s0_open + s1_open > 0:
        md.append(f"> [!CAUTION]\n> **{s0_open + s1_open} BLOCKING FAILURES ACTIVE.** Milestones cannot close.\n")
    else:
        md.append("> [!NOTE]\n> **ZERO BLOCKING FAILURES.** All S0/S1 failures are resolved.\n")
        
    md.extend([
        "## Summary by Severity",
        "",
        "| Severity | Total | Active | Closed |",
        "|---|---|---|---|"
    ])
    for sev in ["S0", "S1", "S2", "S3"]:
        tot = sum(1 for r in records if r["severity"] == sev)
        act = sum(1 for r in records if r["severity"] == sev and r["status"] != "CLOSED")
        cls = tot - act
        md.append(f"| {sev} | {tot} | {act} | {cls} |")
        
    md.extend([
        "",
        "## Detailed Records",
        "",
        "| ID | Milestone | Sev | Status | Cat | Esc | Symptom | Detected By |",
        "|---|---|---|---|---|---|---|---|"
    ])
    
    if not records:
        md.append("| - | - | - | - | - | - | *No failures logged.* | - |")
    else:
        for r in records:
            md.append(
                f"| `{r['id']}` | {r['milestone']} | **{r['severity']}** | `{r['status']}` | "
                f"{r['category']} | L{r['escalation_level']} | {r['symptom']} | {r['detected_by']} |"
            )
            
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
