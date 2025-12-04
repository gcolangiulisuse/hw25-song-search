#!/bin/bash

# Download CLAP model if not present
MODEL_PATH="/app/config/music_audioset_epoch_15_esc_90.14.pt"

if [ ! -f "$MODEL_PATH" ]; then
    echo "📥 Downloading CLAP model (one-time, ~2.35 GB)..."
    mkdir -p /app/config
    wget -O "$MODEL_PATH" https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt
    echo "✅ Model downloaded!"
else
    echo "✅ Model already exists"
fi

# Start the application
python /app/backend/main.py
