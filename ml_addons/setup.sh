#!/bin/bash
# Quick Start Script - AI-Driven Adaptive Modulation
# Bold (Unubold) - EEME30003 Major Project

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║   AI-Driven Adaptive Modulation - Quick Start Setup          ║"
echo "║   Bold (Unubold) - EEME30003 Major Project                   ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Create virtual environment
echo "[1/4] Creating Python virtual environment..."
if [ -d "venv" ]; then
    echo "  ℹ Virtual environment already exists, skipping..."
else
    python3 -m venv venv
    echo "  ✓ Virtual environment created"
fi

# Step 2: Activate virtual environment
echo ""
echo "[2/4] Activating virtual environment..."
source venv/bin/activate
echo "  ✓ Virtual environment activated"

# Step 3: Install Python dependencies
echo ""
echo "[3/4] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "  ✓ Dependencies installed"

# Step 4: Verify installation
echo ""
echo "[4/4] Verifying installation..."
python3 << END
import torch
import numpy as np
import pandas as pd
import sklearn
import matplotlib
print("  ✓ PyTorch:", torch.__version__)
print("  ✓ NumPy:", np.__version__)
print("  ✓ Pandas:", pd.__version__)
print("  ✓ Scikit-learn:", sklearn.__version__)
print("  ✓ Matplotlib:", matplotlib.__version__)
END

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete!                            ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Next Steps:"
echo "  1. Install MATLAB Engine for Python"
echo "     In MATLAB: cd (fullfile(matlabroot,'extern','engines','python'))"
echo "                system('python setup.py install')"
echo "  2. Generate training data: matlab -batch 'run generate_training_data.m'"
echo "  3. Train AI agent: python feedback_agent.py"
echo "  4. Run integration: python matlab_ai_integration.py"
echo ""
