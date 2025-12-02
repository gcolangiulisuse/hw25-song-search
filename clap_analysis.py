#!/usr/bin/env python3
"""
CLAP Analysis Module
Handles all CLAP-specific logic for audio analysis with full song coverage.
Uses parallel processing to analyze segments across multiple CPU cores.
OPTIMIZED: Analyzes each song once, then compares against all queries.
"""

import os
import sys

# CRITICAL: Suppress ALL warnings BEFORE importing any other modules
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import warnings
warnings.filterwarnings('ignore')

# Suppress transformers logging BEFORE importing
import logging
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

import random
import numpy as np
import torch
import laion_clap
import librosa
import time
from multiprocessing import Pool, cpu_count
from functools import partial

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


# Global model instance for multiprocessing workers
_global_model = None

def _init_worker_model(model_path):
    """Initialize model in each worker process (silently)."""
    global _global_model
    if _global_model is None:
        # Suppress ALL output in worker process
        import os
        import sys
        import warnings
        import logging
        
        # Set environment variables
        os.environ['PYTHONWARNINGS'] = 'ignore'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Suppress warnings
        warnings.filterwarnings('ignore')
        
        # Suppress transformers logging
        logging.getLogger('transformers').setLevel(logging.ERROR)
        logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)
        logging.getLogger().setLevel(logging.ERROR)
        
        # Redirect stdout AND stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        try:
            import laion_clap
            _global_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
            _global_model.load_ckpt(model_path)
            _global_model.eval()
        finally:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        # Set seeds in worker process
        import random
        import numpy as np
        import torch
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)


def _process_segment(segment):
    """
    Process a single segment in a worker process.
    Returns the audio embedding for this segment.
    
    Args:
        segment: Raw audio segment (1D numpy array)
    
    Returns:
        numpy array: Audio embedding
    """
    # Reshape for model input
    segment_input = segment.reshape(1, -1)
    
    # Get audio embedding (CLAP converts RAW audio to spectrogram internally)
    with torch.no_grad():
        audio_embedding = _global_model.get_audio_embedding_from_data(
            x=segment_input,
            use_tensor=False
        )
    
    return audio_embedding[0]  # Return the embedding vector


class CLAPAnalyzer:
    """
    CLAP-based audio analyzer with full song coverage using overlapping segments.
    Uses parallel processing to analyze segments across all CPU cores.
    OPTIMIZED: Analyzes each song once, then compares against multiple queries.
    """
    
    def __init__(self, segment_length=480000, hop_length=240000, num_workers=None):
        """
        Initialize CLAP analyzer.
        
        Args:
            segment_length: Length of each segment in samples (480000 = 10 sec at 48kHz)
            hop_length: Distance between segment starts (240000 = 5 sec at 48kHz = 50% overlap)
            num_workers: Number of parallel workers (None = auto-detect CPU cores)
        """
        self.segment_length = segment_length
        self.hop_length = hop_length
        self.sample_rate = 48000
        self.model = None
        self.model_path = None
        
        # Auto-detect CPU cores
        if num_workers is None:
            self.num_workers = cpu_count()
        else:
            self.num_workers = num_workers
        
    def initialize_model(self):
        """Initialize CLAP model with music-optimized checkpoint."""
        print("🔧 Initializing CLAP model...")
        print(f"   CPU cores: {cpu_count()} | Workers: {self.num_workers}")
        print(f"   Segments: {self.segment_length/self.sample_rate:.0f}s with {100*(1 - self.hop_length/self.segment_length):.0f}% overlap")
        
        # Define model path
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, 'music_audioset_epoch_15_esc_90.14.pt')
        
        # Download model if not present
        if not os.path.exists(self.model_path):
            print(f"\n📥 Downloading model (one-time, ~2.35 GB)...")
            import urllib.request
            url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt'
            
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100.0 / total_size, 100)
                sys.stdout.write(f"\r   Progress: {percent:.1f}% ({downloaded/(1024**3):.2f}/{total_size/(1024**3):.2f} GB)")
                sys.stdout.flush()
            
            urllib.request.urlretrieve(url, self.model_path, reporthook=show_progress)
            print("\n   ✅ Download complete!")
        
        # Load model silently (suppress verbose output)
        print("   Loading model weights...", end='', flush=True)
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        try:
            self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
            self.model.load_ckpt(self.model_path)
            self.model.eval()
        finally:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        print(" ✅")
        print()
        
    def load_audio(self, audio_path):
        """
        Load full audio file at target sample rate.
        Returns RAW AUDIO WAVEFORM (not spectrogram - CLAP does that internally).
        """
        audio_data, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return audio_data
    
    def create_segments(self, audio_data):
        """
        Split RAW audio waveform into overlapping segments.
        
        Args:
            audio_data: Full audio waveform (1D numpy array of raw audio samples)
        
        Returns:
            List of audio segments (each is a 1D numpy array)
        """
        segments = []
        total_length = len(audio_data)
        
        # If audio is shorter than one segment, pad it
        if total_length <= self.segment_length:
            padded = np.pad(audio_data, (0, self.segment_length - total_length), mode='constant')
            return [padded]
        
        # Create overlapping segments by sliding through RAW audio
        for start in range(0, total_length - self.segment_length + 1, self.hop_length):
            segment = audio_data[start:start + self.segment_length]
            segments.append(segment)
        
        # Ensure we don't miss the end of the song
        last_start = len(segments) * self.hop_length
        if last_start < total_length:
            # Take the last segment_length samples
            last_segment = audio_data[-self.segment_length:]
            segments.append(last_segment)
        
        return segments
    
    def analyze_audio_with_queries(self, audio_path, queries):
        """
        OPTIMIZED: Analyze audio file ONCE, then compare against ALL queries.
        This is much faster than calling analyze_audio_with_query() multiple times.
        
        Args:
            audio_path: Path to audio file
            queries: List of text query strings
        
        Returns:
            tuple: (results, avg_embedding) where:
                - results: list of dicts, one per query with:
                    - query: The query text
                    - similarity: Average score across all segments (MAIN METRIC)
                    - max_score: Highest score from any segment
                    - min_score: Lowest score from any segment
                    - std_score: Standard deviation across segments
                    - num_segments: Number of segments analyzed
                    - duration_sec: Audio duration in seconds
                    - analysis_time: Time taken for this query
                - avg_embedding: Average embedding vector for the song (for song-to-song comparison)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        
        print(f"      📊 Analyzing song (loading + {len(queries)} queries)...", end='', flush=True)
        total_start = time.time()
        
        # STEP 1: Load audio and create segments (ONCE)
        audio_data = self.load_audio(audio_path)
        duration_sec = len(audio_data) / self.sample_rate
        segments = self.create_segments(audio_data)
        
        # STEP 2: Get audio embeddings for all segments in parallel (ONCE)
        with Pool(processes=self.num_workers, 
                  initializer=_init_worker_model, 
                  initargs=(self.model_path,)) as pool:
            audio_embeddings = pool.map(_process_segment, segments)
        
        # Convert to numpy array for efficient computation
        audio_embeddings = np.array(audio_embeddings)
        
        # Compute average embedding for song-to-song comparison
        avg_embedding = np.mean(audio_embeddings, axis=0)
        # Normalize to unit length (for cosine similarity via dot product)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        load_elapsed = time.time() - total_start
        print(f" ✅ ({load_elapsed:.2f}s)")
        
        # STEP 3: Compare against each query (fast, no audio processing)
        results = []
        for query_idx, query in enumerate(queries, 1):
            query_start = time.time()
            
            print(f"      [{query_idx}/{len(queries)}] Comparing: \"{query}\"", end='', flush=True)
            
            # Get text embedding for this query
            with torch.no_grad():
                text_embedding = self.model.get_text_embedding([query], use_tensor=False)
            
            # Compute similarities with all audio segments (vectorized)
            similarities = np.dot(audio_embeddings, text_embedding[0])
            
            query_elapsed = time.time() - query_start
            avg_score = np.mean(similarities)
            
            # Get label with icon (with explicit space after icon)
            label = get_similarity_label(avg_score)
            icon = get_similarity_icon(avg_score)
            
            print(f" → {avg_score:.4f} {icon} {label} ({query_elapsed:.2f}s)")
            
            results.append({
                'query': query,
                'similarity': avg_score,
                'max_score': float(np.max(similarities)),
                'min_score': float(np.min(similarities)),
                'std_score': float(np.std(similarities)),
                'num_segments': len(segments),
                'duration_sec': duration_sec,
                'analysis_time': query_elapsed
            })
        
        return results, avg_embedding


def get_similarity_label(score):
    """Convert similarity score to human-readable label."""
    if score > 0.3:
        return "HIGH"
    elif score > 0.15:
        return "MODERATE"
    else:
        return "LOW"


def get_similarity_icon(score):
    """Get icon for similarity score."""
    if score > 0.3:
        return "✅"
    elif score > 0.15:
        return "⚠️ "
    else:
        return "❌"
