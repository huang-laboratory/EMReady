"""Top-level EMReady CLI dispatcher."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "ligand":
        from emready.commands.ligand import main as ligand_main

        return ligand_main(argv[1:])
    from emready.commands.predict import main as predict_main

    return predict_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
