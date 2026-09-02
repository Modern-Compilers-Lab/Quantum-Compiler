"""Resolve which CSV a render script reads: freshly generated, or committed."""

from __future__ import annotations

from pathlib import Path

from common import OUTPUT_DIR, SUMMARY_ROOT

GENERATED_ROOT = OUTPUT_DIR / "csv"

_mode = "auto"
_generated_root = GENERATED_ROOT
_used: list[tuple[str, str]] = []


def configure(mode="auto", generated_root=None):
    global _mode, _generated_root
    _mode = mode
    _used.clear()
    if generated_root:
        _generated_root = Path(generated_root)


def used():
    """Every (rel, origin) resolved since the last configure(), in order."""
    return list(_used)


def _record(rel, origin):
    _used.append((str(rel), origin))


def resolve(rel, hint=""):
    """Return (path, origin) for a CSV given its path relative to the CSV root."""
    gen = _generated_root / rel
    com = SUMMARY_ROOT / rel
    if _mode == "generated":
        if not gen.exists():
            raise SystemExit(_missing(rel, gen, hint, only="generated"))
        _record(rel, "generated")
        return gen, "generated"
    if _mode == "committed":
        if not com.exists():
            raise SystemExit(_missing(rel, com, hint, only="committed"))
        _record(rel, "committed")
        return com, "committed"
    if gen.exists():
        _record(rel, "generated")
        return gen, "generated"
    if com.exists():
        _record(rel, "committed")
        return com, "committed"
    raise SystemExit(_missing(rel, gen, hint))


def _missing(rel, path, hint, only=None):
    msg = f"CSV not found: {rel}\n    looked at {path}"
    if only is None:
        msg += f"\n    and at        {SUMMARY_ROOT / rel}"
    if hint:
        msg += f"\n\n  Generate it with:\n      {hint}"
    return msg
