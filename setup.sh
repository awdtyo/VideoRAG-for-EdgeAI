#!/usr/bin/env bash
# Run this on the Raspberry Pi 5 itself (or your Arch dev machine to test first).
set -e

echo "=== System packages ==="
# Arch Linux (your Pi setup, per its README). Adjust for other distros.
sudo pacman -Sy --needed ffmpeg tesseract tesseract-data-eng python python-pip

echo "=== Python virtual environment ==="
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Ollama ==="
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Install it: curl -fsSL https://ollama.com/install.sh | sh"
else
    echo "Ollama already installed."
fi
echo "Pull the generation model with: ollama pull qwen2.5:3b"

echo "=== Model weights ==="
bash download_models.sh

echo ""
echo "Setup complete. Activate the venv with: source .venv/bin/activate"
echo "Then: python main.py ingest data/videos/my_clip.mp4"
echo "      python main.py query \"what happened in the video?\""
