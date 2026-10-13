#!/usr/bin/env python3
"""Check the handoff's SHA-256 manifest (standard library only).
Usage: python tools/verify_manifest.py [root] [--manifest DEVELOPMENT_MANIFEST.sha256]
"""
from __future__ import annotations
import hashlib
import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--manifest', default='MANIFEST.sha256')
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = root / args.manifest
    if not manifest.is_file():
        print(f"ERROR: no manifest at {manifest}", file=sys.stderr)
        return 2
    failures = []
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            expected, rel = line.split("  ", 1)
            path = (root / rel).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("missing file or path outside root")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                failures.append(f"HASH MISMATCH: {rel}")
            checked += 1
        except (ValueError, OSError) as exc:
            failures.append(f"INVALID ENTRY: {line}: {exc}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        print(f"FAILED: {len(failures)} errors, {checked} files read.")
        return 1
    print(f"OK: {checked} files match SHA-256 manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
