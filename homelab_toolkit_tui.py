#!/usr/bin/env python3
from __future__ import annotations

import sys


def main() -> None:
    try:
        from homelab_tui import HomelabTUI
    except ModuleNotFoundError as exc:
        if exc.name == "textual":
            print("Error: textual is not installed.")
            print("Install it with: python3 -m pip install -r requirements-tui.txt")
            raise SystemExit(1)
        raise

    app = HomelabTUI()
    app.run()


if __name__ == "__main__":
    main()
