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
from datetime import datetime
from clap_analysis import CLAPAnalyzer, get_similarity_label


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
    
    # Process each song (OPTIMIZED: analyze once, compare against all queries)
    for audio_idx, audio_file in enumerate(audio_files, 1):
        filename = os.path.basename(audio_file)
        
        print(f"\n{'='*80}")
        print(f"🎵 [{audio_idx}/{len(audio_files)}] {filename}")
        print(f"{'='*80}")
        
        song_start_time = time.time()
        
        try:
            # OPTIMIZED: Pass all queries at once
            query_results = analyzer.analyze_audio_with_queries(audio_file, queries)
            
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
    
    print(f"\n{'='*80}")
    print(f"✅ Analysis complete!")
    print(f"💾 Results saved to: {csv_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
