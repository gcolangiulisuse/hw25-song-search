"""
Audio analysis module adapted from clap_analysis.py for web app use.
Handles CLAP model initialization and song analysis.
"""

import os
import sys
import glob

# --- CRITICAL FIX: Set Thread Limits BEFORE Imports ---
# These environment variables must be set before numpy/torch are imported.
# If set afterwards, they are often ignored, causing CPU oversubscription
# (e.g., 3 workers * 12 threads each = 36 threads fighting for CPU -> Crash)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
# ------------------------------------------------------

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

import random
import numpy as np
import torch
import laion_clap
import librosa
from multiprocessing import cpu_count
from typing import List, Tuple, Callable, Optional
from mutagen import File as MutagenFile
from mutagen.mp3 import MP3
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_EXCEPTION


# Set all random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ['PYTHONHASHSEED'] = '42'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# --- CRITICAL FIX: Prevent CPU Oversubscription ---
# Ensure PyTorch also respects the single-thread limit at runtime
torch.set_num_threads(1)
# --------------------------------------------------

# Global model for threads (shared memory)
_global_model = None


def _process_segments_batch(segments_batch):
    """Process a batch of segments (runs in a thread)."""
    global _global_model
    if _global_model is None:
        raise RuntimeError("❌ CRITICAL: _global_model is None!")
    
    # Stack all segments into a single batch for faster inference
    # Note: PyTorch releases the GIL during heavy inference, so this runs in parallel!
    batch_input = np.stack(segments_batch, axis=0)
    
    with torch.no_grad():
        audio_embeddings = _global_model.get_audio_embedding_from_data(
            x=batch_input,
            use_tensor=False
        )
    return audio_embeddings


class AudioAnalyzer:
    """Audio analyzer for web app."""
    
    def __init__(self, model_path: str = "/app/config/music_audioset_epoch_15_esc_90.14.pt",
                 segment_length: int = 480000, hop_length: int = 240000, 
                 num_workers: Optional[int] = None):
        """
        Initialize audio analyzer.
        
        Args:
            model_path: Path to CLAP model checkpoint
            segment_length: Segment length in samples (10s at 48kHz)
            hop_length: Hop length in samples (5s at 48kHz = 50% overlap)
            num_workers: Number of parallel workers
        """
        self.model_path = model_path
        self.segment_length = segment_length
        self.hop_length = hop_length
        self.sample_rate = 48000
        self.model = None
        self.executor = None  # ThreadPoolExecutor
        
        if num_workers is None:
            # Check for env var override
            env_workers = os.environ.get('MAX_ANALYSIS_WORKERS')
            if env_workers:
                self.num_workers = int(env_workers)
                print(f"🔧 Using configured workers from env: {self.num_workers}")
            else:
                total_cores = cpu_count()
                # CRITICAL FIX: Cap max workers to prevent OOM kills.
                # Even if you have 12 cores, running 10 LLM-inference threads usually kills RAM.
                # We cap at 3 by default to be safe.
                safe_limit = 10
                self.num_workers = max(1, min(safe_limit, total_cores - 1))
                print(f"🔧 Auto-configured workers: {self.num_workers} (Safe mode: max {safe_limit} threads to prevent crash)")
        else:
            self.num_workers = num_workers
        
        print(f"🔧 AudioAnalyzer initialized with {self.num_workers} threads")
    
    def initialize_model(self):
        """Initialize CLAP model."""
        if self.model is not None:
            return  # Already initialized
        
        print(f"🔧 Initializing CLAP model from {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        # Load model silently
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        
        try:
            self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
            # Patch load_state_dict to ignore unexpected keys
            original_load_state_dict = self.model.model.load_state_dict
            self.model.model.load_state_dict = lambda *args, **kwargs: original_load_state_dict(*args, **{**kwargs, 'strict': False})
            self.model.load_ckpt(self.model_path)
            self.model.model.load_state_dict = original_load_state_dict
            self.model.eval()
        finally:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        print("✅ Model initialized")
        
        # Load model into global space for threads to access
        global _global_model
        _global_model = self.model
        
        self.text_model = None
    
    def _ensure_pool(self):
        """Ensure thread pool is ready."""
        if self.executor is None:
            print(f"⏱️  Creating fresh thread pool with {self.num_workers} threads...")
            self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
            print(f"✅ Thread pool created")
        else:
            print(f"♻️  Reusing existing thread pool")
    
    def close_pool(self):
        """Close the thread pool."""
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
            print("🔒 Thread pool closed")
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load audio file."""
        audio_data, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return audio_data
    
    def create_segments(self, audio_data: np.ndarray) -> List[np.ndarray]:
        """Create overlapping segments from audio."""
        segments = []
        total_length = len(audio_data)
        
        if total_length <= self.segment_length:
            padded = np.pad(audio_data, (0, self.segment_length - total_length), mode='constant')
            return [padded]
        
        for start in range(0, total_length - self.segment_length + 1, self.hop_length):
            segment = audio_data[start:start + self.segment_length]
            segments.append(segment)
        
        last_start = len(segments) * self.hop_length
        if last_start < total_length:
            last_segment = audio_data[-self.segment_length:]
            segments.append(last_segment)
        
        return segments
    
    def analyze_song(self, audio_path: str, 
                    progress_callback: Optional[Callable[[str], None]] = None) -> Tuple[np.ndarray, float, int]:
        """Analyze a song and return its average embedding."""
        import time
        start_time = time.time()
        print(f"⏱️  Starting analysis of {audio_path}")
        
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        if progress_callback:
            progress_callback("Loading audio...")
        
        # Load audio
        load_start = time.time()
        audio_data = self.load_audio(audio_path)
        duration_sec = len(audio_data) / self.sample_rate
        print(f"⏱️  Audio loaded in {time.time() - load_start:.2f}s ({duration_sec:.1f}s duration)")
        
        if progress_callback:
            progress_callback("Creating segments...")
        
        seg_start = time.time()
        segments = self.create_segments(audio_data)
        print(f"⏱️  Created {len(segments)} segments in {time.time() - seg_start:.2f}s")
        
        if progress_callback:
            progress_callback(f"Analyzing {len(segments)} segments...")
        
        # Ensure pool is ready
        self._ensure_pool()
        
        # --- MEMORY FIX: Use fixed small batch sizes ---
        # Previous logic divided song into N chunks (where N=workers).
        # For long songs (e.g. 10 mins), this created massive batches (e.g. 50+ segments)
        # which caused OOM when running in parallel.
        # We now use a fixed, small batch size.
        BATCH_SIZE = 4 
        
        segment_batches = []
        for i in range(0, len(segments), BATCH_SIZE):
            segment_batches.append(segments[i:i + BATCH_SIZE])
        
        print(f"⏱️  Processing {len(segments)} segments in {len(segment_batches)} batches (size {BATCH_SIZE}) with thread pool...")
        map_start = time.time()
        
        try:
            # 2 minute timeout
            timeout = 120
            
            # Submit all batches
            # The executor (size=3) will handle the queueing, keeping memory usage to (3 * BATCH_SIZE)
            futures = [self.executor.submit(_process_segments_batch, batch) for batch in segment_batches]
            
            # Wait for results
            done, not_done = wait(futures, timeout=timeout, return_when=FIRST_EXCEPTION)
            
            if not_done:
                # Timeout occurred
                for future in not_done:
                    future.cancel()
                # We don't necessarily need to kill the executor in threading (unlike process pool)
                # but it's safer to restart it if something got stuck.
                self.executor.shutdown(wait=False)
                self.executor = None
                raise RuntimeError(f"Analysis timed out after {timeout}s.")
            
            # Get results
            batch_results = [future.result() for future in futures]
            audio_embeddings = np.vstack(batch_results)
            print(f"⏱️  Segments processed in {time.time() - map_start:.2f}s")
            
        except Exception as e:
            # On error, reset pool to be safe
            if self.executor:
                self.executor.shutdown(wait=False)
                self.executor = None
            raise RuntimeError(f"Processing failed: {str(e)}")
        
        # Average and Normalize
        audio_embeddings = np.array(audio_embeddings)
        avg_embedding = np.mean(audio_embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        total_time = time.time() - start_time
        print(f"⏱️  Total analysis time: {total_time:.2f}s")
        
        return avg_embedding, duration_sec, len(segments)
    
    def _ensure_text_model(self):
        """Ensure fresh model for text queries (thread-safe logic)."""
        if self.text_model is None:
            print(f"🔧 Creating fresh model instance for text queries")
            import laion_clap
            
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
            
            try:
                self.text_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
                original_load_state_dict = self.text_model.model.load_state_dict
                self.text_model.model.load_state_dict = lambda *args, **kwargs: original_load_state_dict(*args, **{**kwargs, 'strict': False})
                self.text_model.load_ckpt(self.model_path)
                self.text_model.model.load_state_dict = original_load_state_dict
                self.text_model.eval()
            finally:
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            
            print("✅ Text model ready")
    
    def compute_text_similarity(self, audio_embedding: np.ndarray, query_text: str) -> float:
        """Compute similarity between audio embedding and text query."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        try:
            # Monkey-patch RoBERTa forward to fix input_shape unpacking bug
            import transformers.models.roberta.modeling_roberta as roberta_module
            
            if not hasattr(roberta_module, '_original_roberta_forward'):
                roberta_module._original_roberta_forward = roberta_module.RobertaModel.forward
                
                def patched_forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                                  position_ids=None, head_mask=None, inputs_embeds=None,
                                  encoder_hidden_states=None, encoder_attention_mask=None,
                                  past_key_values=None, use_cache=None, output_attentions=None,
                                  output_hidden_states=None, return_dict=None):
                    if input_ids is not None and input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)
                    if attention_mask is not None and attention_mask.dim() == 1:
                        attention_mask = attention_mask.unsqueeze(0)
                    if token_type_ids is not None and token_type_ids.dim() == 1:
                        token_type_ids = token_type_ids.unsqueeze(0)
                    if inputs_embeds is not None and inputs_embeds.dim() == 2:
                        inputs_embeds = inputs_embeds.unsqueeze(0)
                    
                    return roberta_module._original_roberta_forward(
                        self, input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids, position_ids=position_ids,
                        head_mask=head_mask, inputs_embeds=inputs_embeds,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        past_key_values=past_key_values, use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict
                    )
                
                roberta_module.RobertaModel.forward = patched_forward
                print("🔧 Applied RoBERTa forward() monkey-patch")
            
            with torch.no_grad():
                text_embedding = self.model.get_text_embedding([query_text], use_tensor=False)
            
            if isinstance(text_embedding, np.ndarray):
                if text_embedding.ndim == 2:
                    text_vec = text_embedding[0]
                elif text_embedding.ndim == 1:
                    text_vec = text_embedding
                else:
                    raise ValueError(f"Unexpected shape: {text_embedding.shape}")
            else:
                raise ValueError(f"Unexpected type: {type(text_embedding)}")
            
            similarity = float(np.dot(audio_embedding, text_vec))
            return similarity
            
        except Exception as e:
            print(f"❌ Error in compute_text_similarity: {e}")
            raise
    
    def get_metadata(self, audio_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract artist and title from audio file metadata."""
        try:
            audio_file = MutagenFile(audio_path)
            if audio_file is None:
                return self._parse_filename(audio_path)
            
            artist = None
            title = None
            
            if isinstance(audio_file, MP3):
                artist = audio_file.get('TPE1', [None])[0] if 'TPE1' in audio_file else None
                title = audio_file.get('TIT2', [None])[0] if 'TIT2' in audio_file else None
            elif isinstance(audio_file, (FLAC, MP4)):
                artist = audio_file.get('artist', [None])[0] if 'artist' in audio_file else None
                title = audio_file.get('title', [None])[0] if 'title' in audio_file else None
            else:
                tags = audio_file.tags
                if tags:
                    artist = tags.get('artist', [None])[0] if 'artist' in tags else None
                    title = tags.get('title', [None])[0] if 'title' in tags else None
            
            if isinstance(artist, bytes): artist = artist.decode('utf-8', errors='ignore')
            if isinstance(title, bytes): title = title.decode('utf-8', errors='ignore')
            
            if not artist and not title:
                return self._parse_filename(audio_path)
            
            if not title:
                parsed_artist, parsed_title = self._parse_filename(audio_path)
                title = parsed_title if parsed_title else os.path.splitext(os.path.basename(audio_path))[0]
            
            return artist, title
        except Exception:
            return self._parse_filename(audio_path)
    
    def _parse_filename(self, audio_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse filename to extract artist and title."""
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        for separator in [' - ', ' – ', ' — ', '-']:
            if separator in filename:
                parts = filename.split(separator)
                if len(parts) >= 2:
                    artist = parts[0].strip()
                    if len(parts) == 2:
                        title = parts[1].strip()
                    else:
                        title = separator.join(parts[1:-1]).strip()
                        last_part = parts[-1].strip()
                        if not (len(last_part.split()) == 1 and last_part[0].isupper()):
                            title = separator.join(parts[1:]).strip()
                    return artist, title
        return None, filename


def find_audio_files(directory: str) -> List[str]:
    """Find all audio files in directory."""
    audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.ogg', '*.m4a']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(directory, ext)))
        audio_files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    audio_files = [f for f in audio_files if not os.path.basename(f).startswith('.')]
    return sorted(set(audio_files))