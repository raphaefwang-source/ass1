#!/bin/bash
# Create a virtual environment for the toy runner (run once on a login node).
#   bash hpc/setup_venv.sh VENV_DIR [PYTHON] [--range]
# Default: the pinned versions of requirements.txt (Python >= 3.12). --range: requirements-range.txt
# (version ranges; use it when the pinned versions are unavailable for your Python, e.g. Python 3.10 / 3.11).
set -euo pipefail
VENV="${1:?usage: setup_venv.sh VENV_DIR [PYTHON] [--range]}"
PY="${2:-python3}"
REQ=requirements.txt
[ "${3:-}" = "--range" ] && REQ=requirements-range.txt
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$HERE/$REQ"
"$VENV/bin/python" -c "import sys, numpy, scipy, matplotlib; print(sys.version); print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'matplotlib', matplotlib.__version__)"
echo "PY_SETUP=\"source $VENV/bin/activate\"   # put this line into hpc/cluster.env"
