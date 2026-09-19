#!/usr/bin/env python
"""src/  toarray()/todense()AGENTS.md  + protocol I7

exit 0 = exit 1 =
source ~/sfate_env/bin/activate && python scripts/check_no_dense.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PATTERN = re.compile(r"\.(toarray|todense)\s*\(")
ALLOWLIST = ()  #  'path:line'
SRC = Path("src")


def main() -> int:
    violations: list[str] = []
    for py in sorted(SRC.rglob("*.py")):
        for lineno, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if any(a in f"{py}:{lineno}" for a in ALLOWLIST):
                continue
            if PATTERN.search(line):
                violations.append(f"{py}:{lineno}: {line.strip()}")
    if violations:
        print("[check_no_dense] src/  toarray/todense")
        print("\n".join(violations))
        return 1
    print("[check_no_dense] PASSsrc/  toarray/todense ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
