#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Ensure Python 3.11 is used
PYTHON_VERSION=python3.11

# Check if the specified Python version is installed
if ! command -v $PYTHON_VERSION &> /dev/null; then
  echo "Error: $PYTHON_VERSION is not installed." >&2
  exit 1
fi

# Create virtual environment named 'venv'
# $PYTHON_VERSION -m venv venv
pip install virtualenv
virtualenv venv

# Activate the virtual environment (Windows)
Source .\venv\Scripts\activate

# Upgrade pip, setuptools, and wheel
pip install --upgrade pip setuptools wheel

# Check if requirements.txt exists before attempting to install dependencies
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  echo "Error: requirements.txt not found." >&2
  exit 1
fi

# Install dlib from .whl file if it exists
DLIB_WHL="dlib-19.24.1-cp311-cp311-win_amd64.whl"
if [ -f "$DLIB_WHL" ]; then
  pip install "$DLIB_WHL"
else
  echo "Error: $DLIB_WHL not found." >&2
  exit 1
fi

echo "Setup completed successfully."
