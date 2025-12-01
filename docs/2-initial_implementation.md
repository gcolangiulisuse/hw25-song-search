# Initial Implementation

**Date:** 1 December 2025

## Approach

Single audio file analysis with CLI-based text search using CLAP embeddings.

## Architecture

```
User Input → Load Audio (librosa) → CLAP Model → Audio Embedding
                                                        ↓
User Query → CLAP Model → Text Embedding → Cosine Similarity → Score
```

## Implementation Files

### `clap_analysis.py`
CLI tool for analyzing individual songs and computing similarity with text queries.

**Key Functions:**
- `load_audio_full()` - Loads entire audio file at 48kHz
- `initialize_model()` - Loads CLAP music-optimized model
- `analyze_audio()` - Computes embeddings and similarity score

**Usage:**
```bash
python clap_analysis.py <audio_file> <search_text>
```

### `requirements.txt`
Dependencies:
- `laion-clap` - CLAP framework
- `librosa` - Audio loading
- `torch` - Deep learning backend
- `numpy` - Numerical operations

## Technical Details

**Audio Processing:**
- Loads entire song (no chunking)
- Resamples to 48kHz (CLAP requirement)
- Converts to mono
- Processes full waveform through model

**Model:**
- `music_audioset_epoch_15_esc_90.14.pt` (2.35 GB)
- HTSAT-base audio encoder
- 512-dimensional embeddings
- Auto-downloads on first run

**Similarity Metric:**
- Cosine similarity (dot product of normalized embeddings)
- Range: -1 to 1 (higher = more similar)
- Thresholds: >0.3 HIGH, >0.15 MODERATE, ≤0.15 LOW

## Limitations

1. **Single file processing** - No batch mode yet
2. **No embedding persistence** - Recomputes on every run
3. **No database** - Can't search across multiple songs
4. **Memory usage** - Loads entire audio file in RAM

## Example Usage

```bash
# Activate environment
source venv/bin/activate

# Test with different queries
python clap_analysis.py song.mp3 "upbeat pop song"
python clap_analysis.py song.mp3 "slow jazz piano"
python clap_analysis.py song.mp3 "energetic rock guitar"
```

**Expected Output:**
```
============================================================
CLAP Audio Analysis
============================================================
Audio file: song.mp3
Search text: 'upbeat pop song'
============================================================

Loading audio: song.mp3
Audio loaded: 245.32 seconds, 48000 Hz
Initializing CLAP model...
Model: music_audioset_epoch_15_esc_90.14.pt (music-optimized)
Model loaded successfully!

Analyzing audio...
Computing audio embedding...
Computing text embedding for: 'upbeat pop song'

============================================================
RESULTS
============================================================
Similarity Score: 0.3456

✅ HIGH similarity - Audio matches the text description well
============================================================
```

## Next Steps

1. **Batch processing** - Analyze multiple files at once
2. **Embedding storage** - Save embeddings to avoid recomputation
3. **Database integration** - Store and search across song library
4. **Web interface** - Build UI for easier interaction
5. **Query optimization** - Test different text prompt formats
