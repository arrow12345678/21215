#!/bin/bash
set -e

echo "Starting the setup process for the video editor on Linux..."

# 1. Update package lists and install system dependencies
echo "--------------------------------------------------"
echo "Updating system packages and installing dependencies (python3-tk, python3-venv)..."
echo "You might be asked for your password."
sudo apt-get update
sudo apt-get install -y python3-tk python3-venv ffmpeg

# NOTE: The FFmpeg download section has been removed.
# The 'imageio-ffmpeg' Python package will now provide the executable automatically.
# We are now also installing ffmpeg via apt-get for robustness.

# 2. Create Python virtual environment
echo "--------------------------------------------------"
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' already exists. Skipping creation."
else
    echo "Creating Python virtual environment in '$VENV_DIR'..."
    python3 -m venv $VENV_DIR
fi

# 3. Activate virtual environment and install dependencies from requirements.txt
echo "--------------------------------------------------"
echo "Activating virtual environment and installing Python packages from requirements.txt..."
source $VENV_DIR/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

deactivate

echo "--------------------------------------------------"
echo "✅ Setup complete!"
echo "You can now run the editor using the './run_editor.sh' script."
echo "--------------------------------------------------" 
