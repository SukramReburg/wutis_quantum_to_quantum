#!/usr/bin/env bash
set -e

echo ">>> Using Python interpreter:"
python -c "import sys; print(sys.version)"

echo ">>> Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

echo ">>> Installing requirements..."
pip install -r requirements.txt

echo ">>> Verifying core packages..."
python - << 'EOF'
from qiskit_machine_learning.connectors import TorchConnector
import qiskit, torch

print("Qiskit:", qiskit.__version__)
print("Torch:", torch.__version__)
print("TorchConnector available ✔")
EOF

echo ">>> Installation complete!"
