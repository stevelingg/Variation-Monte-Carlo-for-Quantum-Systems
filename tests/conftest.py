"""Pytest configuration.

Ensures the project root is importable when tests are executed from within the
`tests/` directory (common on Windows/VS Code).
"""

from __future__ import annotations

import sys
from pathlib import Path


# Add repo root (parent of this `tests/` directory) to sys.path so `import vmc`
# resolves to `vmc.py`.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
