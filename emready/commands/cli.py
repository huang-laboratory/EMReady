"""Top-level EMReady CLI entry for the main map-enhancement command."""

from __future__ import annotations

from emready.commands.predict import main as predict_main


def main(argv=None) -> int:
    return predict_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
