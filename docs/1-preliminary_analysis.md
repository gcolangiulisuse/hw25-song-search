# CLAP Preliminary Analysis

**Date:** 1 December 2025

## What is CLAP?

Audio-text embedding framework (like CLIP for audio). Enables text-based song search using natural language.

- **Paper:** [Large-Scale Contrastive Language-Audio Pretraining](https://arxiv.org/abs/2211.06687)
- **Output:** 512-dim embeddings for audio & text
- **Sample Rate:** 48kHz

## Selected Model

**`music_audioset_epoch_15_esc_90.14.pt`** (2.35 GB)

- Best for music (71% GTZAN score vs 51% for alternatives)
- Trained on ~4M music samples
- Zero-shot capability

## Quick Start

```python
import laion_clap

model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
model.load_ckpt('music_audioset_epoch_15_esc_90.14.pt')

audio_embed = model.get_audio_embedding_from_filelist(['song.wav'])
text_embed = model.get_text_embedding(["upbeat pop song"])
similarity = audio_embed @ text_embed.T
```

## Workflow

1. Pre-compute embeddings for all songs (offline)
2. Query: generate text embedding → compute similarity → return top results
3. Performance: ~100-500ms/song, ~10ms/query

## Next Steps

- Download model from [HuggingFace](https://huggingface.co/lukewys/laion_clap)
- Test with sample songs
- Build search interface
