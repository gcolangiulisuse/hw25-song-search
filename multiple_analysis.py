#!/usr/bin/env python3
"""
Multiple Audio Analysis Orchestrator
Simple orchestrator that calls clap_analysis for each song/query combination.
All progress output is handled by clap_analysis.py.
OPTIMIZED: Analyzes each song once, then compares against all queries.
"""

import os
import sys

# CRITICAL: Set environment variables BEFORE any other imports
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import warnings
warnings.filterwarnings('ignore')

import glob
import csv
import time
import numpy as np
from datetime import datetime
from clap_analysis import CLAPAnalyzer, get_similarity_label, get_similarity_icon


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
    
    audio_files = [f for f in audio_files if not os.path.basename(f).startswith('.')]
    
    return sorted(audio_files)


def compute_song_similarities(song_embeddings, song_names):
    """
    Compute pairwise similarities between all songs.
    
    Args:
        song_embeddings: dict mapping song filename to its average embedding
        song_names: list of song filenames
    
    Returns:
        list of dicts with song1, song2, similarity
    """
    similarities = []
    
    for i in range(len(song_names)):
        for j in range(i + 1, len(song_names)):
            song1 = song_names[i]
            song2 = song_names[j]
            
            # Compute cosine similarity (dot product of normalized embeddings)
            sim = float(np.dot(song_embeddings[song1], song_embeddings[song2]))
            
            similarities.append({
                'song1': song1,
                'song2': song2,
                'similarity': sim
            })
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    return similarities


def main():
    """Main orchestration function."""
    
    # Configuration
    songs_dir = os.path.join(os.path.dirname(__file__), 'songs')
    query_file = os.path.join(os.path.dirname(__file__), 'query', 'query.txt')
    
    # Load data
    queries = load_queries(query_file)
    audio_files = load_audio_files(songs_dir)
    
    if not audio_files:
        print(f"❌ No audio files found in {songs_dir}")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = CLAPAnalyzer(segment_length=480000, hop_length=240000)
    analyzer.initialize_model()
    
    results = []
    song_embeddings = {}  # Store average embedding for each song
    
    # Process each song (OPTIMIZED: analyze once, compare against all queries)
    for audio_idx, audio_file in enumerate(audio_files, 1):
        filename = os.path.basename(audio_file)
        
        print(f"\n{'='*80}")
        print(f"🎵 [{audio_idx}/{len(audio_files)}] {filename}")
        print(f"{'='*80}")
        
        song_start_time = time.time()
        
        try:
            # OPTIMIZED: Pass all queries at once (also returns embeddings)
            query_results, avg_embedding = analyzer.analyze_audio_with_queries(audio_file, queries)
            
            # Store song's average embedding for similarity comparison
            song_embeddings[filename] = avg_embedding
            
            # Store results
            for result in query_results:
                results.append({
                    'audio_file': filename,
                    'query': result['query'],
                    'similarity': result['similarity'],
                    'label': get_similarity_label(result['similarity']),
                    'max_score': result['max_score'],
                    'min_score': result['min_score'],
                    'std_score': result['std_score'],
                    'num_segments': result['num_segments'],
                    'duration_sec': result['duration_sec']
                })
        except Exception as e:
            print(f"      ❌ Error processing {filename}: {str(e)}")
            continue
        
        song_elapsed = time.time() - song_start_time
        print(f"\n⏱️  Total time for this song: {song_elapsed:.2f}s")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"results_{timestamp}.csv"
    csv_path = os.path.join(os.path.dirname(__file__), csv_file)
    
    fieldnames = ['audio_file', 'query', 'similarity', 'label', 'max_score', 'min_score', 
                  'std_score', 'num_segments', 'duration_sec']
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    # Compute song-to-song similarities
    if len(song_embeddings) > 1:
        print(f"\n{'='*80}")
        print(f"🎵 🎵 Song Similarities")
        print(f"{'='*80}")
        
        song_sims = compute_song_similarities(song_embeddings, list(song_embeddings.keys()))
        
        # Display top 10 most similar pairs
        print(f"\nTop {min(10, len(song_sims))} most similar song pairs:\n")
        for idx, sim in enumerate(song_sims[:10], 1):
            icon = get_similarity_icon(sim['similarity'])
            print(f"{idx:2d}. {sim['similarity']:.4f} {icon} {sim['song1']}")
            print(f"                   ↔️  {sim['song2']}")
        
        # Save all song similarities to CSV
        similarity_csv = f"results_similarity_{timestamp}.csv"
        similarity_path = os.path.join(os.path.dirname(__file__), similarity_csv)
        
        with open(similarity_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['song1', 'song2', 'similarity', 'label'])
            writer.writeheader()
            for sim in song_sims:
                writer.writerow({
                    'song1': sim['song1'],
                    'song2': sim['song2'],
                    'similarity': sim['similarity'],
                    'label': get_similarity_label(sim['similarity'])
                })
        
        print(f"💾 Song similarities saved to: {similarity_csv}")
    
    print(f"\n{'='*80}")
    print(f"✅ Analysis complete!")
    print(f"💾 Query results saved to: {csv_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
