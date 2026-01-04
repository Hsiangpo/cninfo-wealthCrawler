from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root / "src"))
    from cninfo_ssgs.app.cli import main as _main

    return int(_main())


if __name__ == "__main__":
    raise SystemExit(main())
