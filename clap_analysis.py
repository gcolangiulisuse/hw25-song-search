#!/usr/bin/env python3
"""
CLAP Audio Analysis Tool
Analyzes audio files and computes similarity with text queries using CLAP embeddings.

Usage:
    python clap_analysis.py <audio_file> <search_text>

Example:
    python clap_analysis.py song.mp3 "upbeat pop song"
"""

import sys
import os
import librosa
import numpy as np
import laion_clap
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*NotOpenSSLWarning.*')


def load_audio_full(audio_path, target_sr=48000):
    """
    Load entire audio file with target sample rate.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (CLAP uses 48kHz)
    
    Returns:
        audio_data: Numpy array of audio samples
        sr: Sample rate
    """
    print(f"Loading audio: {audio_path}")
    audio_data, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    duration = len(audio_data) / sr
    print(f"Audio loaded: {duration:.2f} seconds, {sr} Hz")
    return audio_data, sr


def initialize_model():
    """
    Initialize CLAP model with music-optimized checkpoint.
    Downloads model to ./models/ directory if not present.
    
    Returns:
        model: CLAP_Module instance
    """
    print("Initializing CLAP model...")
    print("Model: music_audioset_epoch_15_esc_90.14.pt (music-optimized)")
    
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    
    # Define model path
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'music_audioset_epoch_15_esc_90.14.pt')
    
    # Download model if not present
    if not os.path.exists(model_path):
        print(f"\nDownloading model to {model_path}...")
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
    else:
        print(f"Using cached model from: {model_path}")
    
    # Load the checkpoint
    model.load_ckpt(model_path)
    
    print("Model loaded successfully!")
    return model


def analyze_audio(model, audio_data, search_text):
    """
    Analyze audio and compute similarity with search text.
    
    Args:
        model: CLAP model
        audio_data: Audio samples array
        search_text: Text query to search for
    
    Returns:
        similarity_score: Cosine similarity between audio and text
    """
    print("\nAnalyzing audio...")
    
    # Reshape audio for CLAP (expects batch dimension)
    audio_data = audio_data.reshape(1, -1)
    
    # Get audio embedding from raw audio data
    print("Computing audio embedding...")
    audio_embedding = model.get_audio_embedding_from_data(
        x=audio_data,
        use_tensor=False
    )
    
    # Get text embedding
    print(f"Computing text embedding for: '{search_text}'")
    text_embedding = model.get_text_embedding(
        [search_text],
        use_tensor=False
    )
    
    # Compute cosine similarity (embeddings are normalized)
    similarity = np.dot(audio_embedding[0], text_embedding[0])
    
    return similarity


def main():
    """Main function to handle CLI arguments and run analysis."""
    
    # Check arguments
    if len(sys.argv) != 3:
        print("Usage: python clap_analysis.py <audio_file> <search_text>")
        print("\nExample:")
        print('  python clap_analysis.py song.mp3 "upbeat pop song"')
        sys.exit(1)
    
    audio_file = sys.argv[1]
    search_text = sys.argv[2]
    
    # Validate audio file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)
    
    print("=" * 60)
    print("CLAP Audio Analysis")
    print("=" * 60)
    print(f"Audio file: {audio_file}")
    print(f"Search text: '{search_text}'")
    print("=" * 60)
    print()
    
    try:
        # Load the entire audio file
        audio_data, sr = load_audio_full(audio_file)
        
        # Initialize CLAP model
        model = initialize_model()
        
        # Analyze audio and compute similarity
        similarity_score = analyze_audio(model, audio_data, search_text)
        
        # Display results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Similarity Score: {similarity_score:.4f}")
        print()
        
        # Interpret the score
        if similarity_score > 0.3:
            print("✅ HIGH similarity - Audio matches the text description well")
        elif similarity_score > 0.15:
            print("⚠️  MODERATE similarity - Audio partially matches the description")
        else:
            print("❌ LOW similarity - Audio doesn't match the description")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
