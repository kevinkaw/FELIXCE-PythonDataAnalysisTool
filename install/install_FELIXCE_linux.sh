#!/bin/bash

ENV_FILE="environment.yml"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
INSTALL_PATH="$HOME/miniconda3"

echo "--- Checking for Conda ---"

# 1. Check if conda is already in the PATH
if command -v conda &> /dev/null; then
    echo "Conda is already installed and in your PATH."
    CONDA_PATH=$(which conda)
# 2. Check common manual install locations
elif [ -d "$HOME/anaconda3" ]; then
    CONDA_PATH="$HOME/anaconda3/bin/conda"
elif [ -d "$HOME/miniconda3" ]; then
    CONDA_PATH="$HOME/miniconda3/bin/conda"
else
    echo "Conda not found. Installing Miniconda..."
    curl -L $MINICONDA_URL -o miniconda.sh
    bash miniconda.sh -b -p "$INSTALL_PATH"
    rm miniconda.sh
    CONDA_PATH="$INSTALL_PATH/bin/conda"
fi

# 3. Use the discovered/installed conda to activate
CONDA_BASE=$(dirname $(dirname "$CONDA_PATH"))
source "$CONDA_BASE/bin/activate"

# 4. Create environment
# Check if environment already exists
if conda info --envs | grep -q "FELIXCE_v2026.02.12"; then
    echo "Environment exists. Updating and pruning..."
    conda env update -f "$ENV_FILE" --prune
else
    echo "Creating new environment..."
    conda env create -f "$ENV_FILE"
fi

echo "--- Setup Complete ---"