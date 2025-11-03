#!/bin/bash
# Simple script to run SimMIM pretraining
# Usage: bash scripts/pretrain.sh

set -e  # Exit on error

echo "Starting SimMIM pretraining..."
echo "=============================="
echo ""

# Run pretraining with default config
python pretrain.py

echo ""
echo "Pretraining complete!"
