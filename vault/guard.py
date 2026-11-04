"""
Cryptographic Vault Guard for Project LEAN.

Prevents premature evaluation and data contamination on held-out test splits (T1, T2, T3)
prior to official, immutable preregistration and locking.
"""

import os
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional

VAULT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = VAULT_DIR / "manifest.json"
PREREG_DIR = VAULT_DIR.parent / "prereg"

LOCKED_SPLITS = {"test", "t1", "t2", "t3", "locked"}
OPEN_SPLITS = {"calibration", "train", "dev", "val"}


class VaultLockViolation(PermissionError):
    """Raised when an attempt is made to access locked test data without authorization."""
    pass


class VaultIntegrityError(ValueError):
    """Raised when dataset content does not match recorded SHA-256 in manifest."""
    pass


def compute_sha256(filepath: Path) -> str:
    """Compute standard SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_active_lock_hashes() -> Dict[str, str]:
    """Retrieve SHA-256 hashes of all committed LOCK files in prereg/."""
    lock_hashes = {}
    if not PREREG_DIR.exists():
        return lock_hashes
    
    for lock_file in PREREG_DIR.glob("LOCK_v*.md"):
        lock_hashes[lock_file.name] = compute_sha256(lock_file)
    return lock_hashes


def verify_unlock_token(token: Optional[str]) -> bool:
    """Verify if the provided token matches any committed LOCK file SHA-256."""
    if not token:
        return False
    active_locks = get_active_lock_hashes()
    return token.strip() in active_locks.values()


def check_split_permission(split_name: str) -> None:
    """
    Enforce vault lock: open splits allowed; locked splits require valid LEAN_UNLOCK token.
    """
    normalized_split = split_name.lower().strip()
    if normalized_split in LOCKED_SPLITS:
        token = os.environ.get("LEAN_UNLOCK")
        if not token:
            raise VaultLockViolation(
                f"VAULT LOCKED: Access to held-out test split '{split_name}' is strictly forbidden. "
                "Set environment variable LEAN_UNLOCK to the SHA-256 hash of a committed prereg/LOCK_v*.md."
            )
        if not verify_unlock_token(token):
            raise VaultLockViolation(
                f"VAULT LOCKED: Invalid LEAN_UNLOCK token '{token}'. "
                "Token must match the exact SHA-256 hash of a committed prereg/LOCK_v*.md file."
            )


def load_manifest() -> Dict[str, Any]:
    """Load the vault manifest."""
    if not MANIFEST_PATH.exists():
        return {"files": {}}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest_data: Dict[str, Any]) -> None:
    """Save the vault manifest."""
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)


def register_vault_file(relative_path: str, split: str) -> str:
    """
    Register a data file into the vault manifest with its SHA-256 hash.
    """
    file_path = VAULT_DIR / relative_path
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot register non-existent file: {file_path}")
    
    file_hash = compute_sha256(file_path)
    manifest = load_manifest()
    manifest["files"][relative_path] = {
        "sha256": file_hash,
        "split": split,
        "size_bytes": file_path.stat().st_size
    }
    save_manifest(manifest)
    return file_hash


def deregister_vault_file(relative_path: str) -> None:
    """Remove a file from the vault manifest."""
    manifest = load_manifest()
    if "files" in manifest and relative_path in manifest["files"]:
        del manifest["files"][relative_path]
        save_manifest(manifest)


def open_vault_file(relative_path: str, mode: str = "r") -> Any:
    """
    Safely access a vault file subject to lock checks and SHA-256 integrity verification.
    """
    manifest = load_manifest()
    file_meta = manifest.get("files", {}).get(relative_path)
    
    if not file_meta:
        raise FileNotFoundError(f"File '{relative_path}' is not registered in vault manifest.")
        
    check_split_permission(file_meta["split"])
    
    full_path = VAULT_DIR / relative_path
    if not full_path.exists():
        raise FileNotFoundError(f"Vault file missing on disk: {full_path}")
        
    current_hash = compute_sha256(full_path)
    if current_hash != file_meta["sha256"]:
        raise VaultIntegrityError(
            f"TAMPER DETECTED: File '{relative_path}' hash {current_hash} does not match "
            f"manifest hash {file_meta['sha256']}."
        )
        
    return open(full_path, mode, encoding="utf-8" if "b" not in mode else None)
