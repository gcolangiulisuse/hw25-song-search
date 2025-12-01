#!/usr/bin/env python3
"""
CLAP Multiple Audio Analysis Tool
Analyzes multiple audio files against multiple text queries using CLAP embeddings.

Usage:
    python multiple_analysis.py

Reads:
    - Audio files from ./songs/ directory
    - Text queries from ./query/query.txt (one query per line)

Output:
    - Results table showing similarity scores for all combinations
    - CSV file with detailed results
"""

import sys
import os
import glob
import random
import librosa
import numpy as np
import torch
import laion_clap
import warnings
from datetime import datetime
import csv

# ============================================================================
# CRITICAL: Set ALL random seeds for complete reproducibility
# ============================================================================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Additional PyTorch determinism settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

# Set environment variables for complete determinism
os.environ['PYTHONHASHSEED'] = '42'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*NotOpenSSLWarning.*')


def load_audio_full(audio_path, target_sr=48000):
    """Load entire audio file with target sample rate."""
    audio_data, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    return audio_data


def initialize_model():
    """Initialize CLAP model with music-optimized checkpoint."""
    print("Initializing CLAP model...")
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    
    # Define model path
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'music_audioset_epoch_15_esc_90.14.pt')
    
    # Download model if not present
    if not os.path.exists(model_path):
        print(f"Downloading model to {model_path}...")
        print("This is a one-time download (~2.35 GB). Please wait...")
        import urllib.request
        url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt'
        
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100.0 / total_size, 100)
            sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded/(1024**3):.2f}/{total_size/(1024**3):.2f} GB)")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(url, model_path, reporthook=show_progress)
        print("\n✅ Model downloaded successfully!")
    
    model.load_ckpt(model_path)
    print("✅ Model loaded successfully!\n")
    return model


def load_queries(query_file):
    """Load text queries from file (one per line)."""
    if not os.path.exists(query_file):
        print(f"❌ Error: Query file not found: {query_file}")
        sys.exit(1)
    
    with open(query_file, 'r') as f:
        queries = [line.strip() for line in f if line.strip()]
    
    return queries


def load_audio_files(songs_dir):
    """Load all audio files from directory."""
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.ogg', '*.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(songs_dir, ext)))
    
    # Filter out hidden files and directories
    audio_files = [f for f in audio_files if not os.path.basename(f).startswith('.')]
    
    return sorted(audio_files)


def get_similarity_label(score):
    """Convert similarity score to label."""
    if score > 0.3:
        return "HIGH"
    elif score > 0.15:
        return "MODERATE"
    else:
        return "LOW"


def main():
    """Main function to run batch analysis."""
    
    # Configuration
    songs_dir = os.path.join(os.path.dirname(__file__), 'songs')
    query_file = os.path.join(os.path.dirname(__file__), 'query', 'query.txt')
    
    print("=" * 80)
    print("CLAP MULTIPLE AUDIO ANALYSIS")
    print("=" * 80)
    print()
    
    # Load queries
    print("📝 Loading queries...")
    queries = load_queries(query_file)
    print(f"   Found {len(queries)} queries:")
    for i, query in enumerate(queries, 1):
        print(f"   {i}. \"{query}\"")
    print()
    
    # Load audio files
    print("🎵 Loading audio files...")
    audio_files = load_audio_files(songs_dir)
    if not audio_files:
        print(f"❌ No audio files found in {songs_dir}")
        sys.exit(1)
    
    print(f"   Found {len(audio_files)} audio files:")
    for i, audio_file in enumerate(audio_files, 1):
        filename = os.path.basename(audio_file)
        print(f"   {i}. {filename}")
    print()
    
    # Initialize model
    model = initialize_model()
    
    # Set model to eval mode for determinism
    model.eval()
    
    # Prepare results storage
    results = []
    
    # Process each audio file
    print("🔄 Processing audio files...\n")
    
    # Disable gradient computation for inference (improves determinism)
    with torch.no_grad():
        for audio_idx, audio_file in enumerate(audio_files, 1):
            filename = os.path.basename(audio_file)
            print(f"[{audio_idx}/{len(audio_files)}] Processing: {filename}")
            
            # Load audio
            try:
                audio_data = load_audio_full(audio_file)
                audio_data = audio_data.reshape(1, -1)
                
                # Get audio embedding
                audio_embedding = model.get_audio_embedding_from_data(
                    x=audio_data,
                    use_tensor=False
                )
                
                # Test against all queries
                for query in queries:
                    # Get text embedding
                    text_embedding = model.get_text_embedding(
                        [query],
                        use_tensor=False
                    )
                    
                    # Compute similarity
                    similarity = float(np.dot(audio_embedding[0], text_embedding[0]))
                    label = get_similarity_label(similarity)
                    
                    results.append({
                        'audio_file': filename,
                        'query': query,
                        'similarity': similarity,
                        'label': label
                    })
                
                print(f"   ✅ Completed\n")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}\n")
                continue
    
    # Display results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    # Group results by audio file
    for audio_file in audio_files:
        filename = os.path.basename(audio_file)
        print(f"🎵 {filename}")
        print("-" * 80)
        
        file_results = [r for r in results if r['audio_file'] == filename]
        for result in file_results:
            score = result['similarity']
            label = result['label']
            query = result['query']
            
            # Color coding
            if label == "HIGH":
                icon = "✅"
            elif label == "MODERATE":
                icon = "⚠️ "
            else:
                icon = "❌"
            
            print(f"  {icon} {label:8s} | {score:6.4f} | \"{query}\"")
        print()
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"results_{timestamp}.csv"
    csv_path = os.path.join(os.path.dirname(__file__), csv_file)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['audio_file', 'query', 'similarity', 'label'])
        writer.writeheader()
        writer.writerows(results)
    
    print("=" * 80)
    print(f"💾 Results saved to: {csv_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
