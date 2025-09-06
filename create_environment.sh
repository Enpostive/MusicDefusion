#!/bin/bash

# Exit on error
set -e

# Environment name
ENV_NAME="mlxmidichords"

echo "🔧 Creating Python 3.10 virtual environment..."
python3.10 -m venv $ENV_NAME
source $ENV_NAME/bin/activate

echo "📦 Upgrading pip..."
pip install --upgrade pip

echo "🚀 Installing MLX..."
pip install mlx

echo "📚 Installing other required libraries..."
pip install numpy tqdm matplotlib sentence-transformers pretty_midi python-Levenshtein empath

# Optional: If needed, sqlite3 bindings (usually already included)
# pip install pysqlite3-binary

echo "✅ Environment setup complete!"
echo "👉 To activate this environment later, run:"
echo "   source $ENV_NAME/bin/activate"

