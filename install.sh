#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.12}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter '$PYTHON_BIN' was not found."
  exit 1
fi

echo ">>> Creating virtual environment at $VENV_DIR"
rm -rf "$VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

VENV_PYTHON="$VENV_DIR/bin/python"

echo ">>> Using Python interpreter:"
"$VENV_PYTHON" -c "import sys; print(sys.version)"

echo ">>> Upgrading pip..."
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

echo ">>> Installing requirements..."
"$VENV_PYTHON" -m pip install -r requirements.txt

echo ">>> Verifying core packages..."
"$VENV_PYTHON" - << 'EOF'
from qiskit_machine_learning.connectors import TorchConnector
import qiskit
import qiskit_aer
import torch
import yaml
import numpy
import pandas
import scipy
import sklearn

print("Qiskit:", qiskit.__version__)
print("Qiskit Aer:", qiskit_aer.__version__)
print("Torch:", torch.__version__)
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("SciPy:", scipy.__version__)
print("scikit-learn:", sklearn.__version__)
print("PyYAML:", yaml.__version__)
print("TorchConnector available ✔")
EOF

echo
echo ">>> Installation complete."
echo ">>> Activate the environment with:"
echo "source $VENV_DIR/bin/activate"