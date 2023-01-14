"""
Entry point for command line interface via `python -m <prog>`.
"""

from .cli import console_entry_point

if __name__ == "__main__":
    raise SystemExit(console_entry_point())
