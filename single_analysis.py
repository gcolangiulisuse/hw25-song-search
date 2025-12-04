#!/usr/bin/env python3
"""
Single Audio Analysis CLI (Offline Tool - NOT used by web interface)
Analyzes one audio file against one text query using CLAP.

NOTE: This is a standalone CLI tool for offline analysis.
The web interface uses backend/analysis.py instead.
"""

import os
import sys
import argparse

# CRITICAL: Set environment variables BEFORE any other imports
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import warnings
warnings.filterwarnings('ignore')

from clap_analysis import CLAPAnalyzer, get_similarity_label, get_similarity_icon


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Analyze audio file against text query using CLAP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s song.mp3 "electric guitar"
  %(prog)s "./songs/Classical Piano.mp3" "piano music"
        '''
    )
    
    parser.add_argument('audio_file', help='Path to audio file (mp3, wav, flac, ogg, m4a)')
    parser.add_argument('query', help='Text query to match against audio')
    
    args = parser.parse_args()
    
    # Validate audio file
    if not os.path.exists(args.audio_file):
        print(f"❌ Error: Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    # Print header
    print("\n" + "="*80)
    print("CLAP Audio Analysis")
    print("="*80)
    print(f"Audio file: {args.audio_file}")
    print(f"Search text: '{args.query}'")
    print("="*80 + "\n")
    
    # Initialize analyzer
    analyzer = CLAPAnalyzer(segment_length=480000, hop_length=240000)
    analyzer.initialize_model()
    
    # Analyze
    print(f"🔍 Analyzing audio against query...\n")
    
    try:
        results, _ = analyzer.analyze_audio_with_queries(args.audio_file, [args.query])
        result = results[0]
        
        # Print results
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"Similarity Score: {result['similarity']:.4f}")
        print(f"Duration: {result['duration_sec']:.2f}s")
        print(f"Segments analyzed: {result['num_segments']}")
        print()
        
        # Get classification
        label = get_similarity_label(result['similarity'])
        icon = get_similarity_icon(result['similarity'])
        
        if label == "HIGH":
            print(f"{icon} HIGH similarity - Audio matches the text description well")
        elif label == "MODERATE":
            print(f"{icon} MODERATE similarity - Audio partially matches the text description")
        else:
            print(f"{icon} LOW similarity - Audio does not match the text description")
        
        print("="*80 + "\n")
        
        # Print detailed stats
        print("Detailed Statistics:")
        print(f"  Average score: {result['similarity']:.4f}")
        print(f"  Maximum score: {result['max_score']:.4f}")
        print(f"  Minimum score: {result['min_score']:.4f}")
        print(f"  Std deviation: {result['std_score']:.4f}")
        print()
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
