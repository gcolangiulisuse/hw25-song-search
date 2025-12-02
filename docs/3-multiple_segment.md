# Multiple Segment Analysis with Overlapping Windows

## Problem
CLAP's `music_audioset_epoch_15_esc_90.14.pt` checkpoint processes maximum **10 seconds** per inference (480,000 samples @ 48kHz). For longer songs, the model's default behavior is to randomly select a 10-second segment (`rand_trunc` mode), which:
- Ignores most of the song content
- Produces non-deterministic results (different segments each run)
- May miss important musical features

## Solution: Overlapping Segment Analysis

### Approach
1. **Split song into overlapping segments**
   - Segment length: 10 seconds (480,000 samples)
   - Hop length: 5 seconds (240,000 samples)
   - **50% overlap** ensures no content is missed

2. **Process each segment independently**
   - Extract CLAP audio embeddings for all segments
   - Parallel processing across all CPU cores

3. **Aggregate results**
   - **AVERAGE similarity score** across all segments (main metric)
   - Also track: max, min, std deviation

### Example
```
Song: 3 minutes = 180 seconds
Segments: 35 overlapping 10-second windows

Timeline:
[0-10s] [5-15s] [10-20s] [15-25s] ... [170-180s]
  └─ 50% overlap ─┘
```

## Performance Optimization

### Architecture
```python
# BEFORE (SLOW - 17x redundant work):
for each song:
    for each of 17 queries:
        - Load audio (17 times!)
        - Create segments (17 times!)
        - Get audio embeddings (17 × N segments!)
        - Compare with text

# AFTER (FAST - analyze once):
for each song:
    - Load audio (1 time)
    - Create segments (1 time)
    - Get audio embeddings (1 × N segments)
    for each of 17 queries:
        - Compare embeddings (fast dot product)
```

### Result
**~15-17x speedup** per song. Audio processing done once, text comparisons are milliseconds.

## Determinism

To ensure **100% reproducible results** across runs:

### 1. Random Seeds
```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

### 2. PyTorch Determinism
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
```

### 3. Environment Variables
```python
os.environ['PYTHONHASHSEED'] = '42'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
```

### 4. Multiprocessing Worker Seeds
Each worker process re-initializes seeds to ensure parallel processing remains deterministic.

### 5. Disabled Model Fusion
```python
enable_fusion=False  # Checkpoint doesn't have fusion layers
```

## Key Implementation Details

- **RAW audio processing**: CLAP receives raw waveform, converts to spectrogram internally
- **Parallel workers**: Each worker loads own model instance (memory efficient)
- **Vectorized comparison**: `np.dot(audio_embeddings, text_embedding)` for fast query matching
- **Clean output**: All warnings suppressed (urllib3, transformers, etc.)

## Validation

Results show meaningful scores:
- Classical piano → **HIGH** scores for "piano", "classical music"
- Classical piano → **LOW** scores for "rock music", "male vocalist"
- Scores are consistent across multiple runs (deterministic)
