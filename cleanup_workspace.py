#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

# One-off helper scripts to remove
FILES_TO_DELETE = [
    ROOT / "create_intl_story_ppt.py",
    ROOT / "create_minimalist_story_ppt.py",
    ROOT / "inspect_metadata.py",
    ROOT / "make_risk_finding_graph.py",
    ROOT / "filter_fr_orig_ge_800.py",
    ROOT / "Supply_Chain_Volatility_Intl.ipynb.bak",
]


def delete_file(path: Path) -> bool:
    try:
        if path.exists() and path.is_file():
            path.unlink()
            return True
    except (PermissionError, OSError):
        return False
    return False


def delete_old_dashboard_pdfs(out_dir: Path) -> None:
    pdfs = sorted(out_dir.glob("intl_dashboard_full*.pdf"), key=lambda p: p.stat().st_mtime)
    if len(pdfs) <= 1:
        return
    # Keep most recent, delete others
    for p in pdfs[:-1]:
        try:
            p.unlink()
        except (PermissionError, OSError):
            pass


def main() -> int:
    # Remove one-off helper files
    removed = []
    for f in FILES_TO_DELETE:
        if delete_file(f):
            removed.append(str(f.name))

    # Prune old dashboard PDFs, keep the newest
    OUT.mkdir(parents=True, exist_ok=True)
    delete_old_dashboard_pdfs(OUT)

    print("Removed:", ", ".join(removed) if removed else "None")
    print("Cleanup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


