# Song-to-Song Similarity

## Overview

After analyzing songs against text queries, the system computes similarities between songs themselves using their audio embeddings.

## How It Works

### Step 1: Extract Song Embeddings (During Analysis)

Each song is processed once:
```python
# Load audio → Split into 10s segments → Get embeddings for each segment
audio_embeddings = [emb1, emb2, ..., emb10]  # One per segment

# Compute average embedding for the whole song
avg_embedding = np.mean(audio_embeddings, axis=0)

# Normalize to unit length (for cosine similarity via dot product)
avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

# Store for later comparison
song_embeddings[filename] = avg_embedding
```

### Step 2: Compare All Song Pairs

After all songs analyzed:
```python
for song1, song2 in all_pairs:
    # Cosine similarity via dot product (vectors already normalized)
    similarity = np.dot(song_embeddings[song1], song_embeddings[song2])
```

### Step 3: Display Results

```
================================================================================
🎵 🎵 Song Similarities
================================================================================

Top 10 most similar song pairs:

 1. 0.8523 ✅ Classical Piano 1.mp3
                   ↔️  Classical Piano 2.mp3
 2. 0.7891 ✅ Rock Guitar 1.mp3
                   ↔️  Rock Guitar 2.mp3
```

## Key Points

**Same metric as text queries:** Cosine similarity (dot product of normalized embeddings)

**No re-analysis needed:** Uses embeddings already computed during query analysis

**Computational cost:** Negligible (~0.001s for 10 songs = 45 comparisons)

**Use case:** Find similar songs in your collection based on audio content, not metadata

## Example Output

```
🎵 [1/3] Aaron Dunn - Minuet - Classical.mp3
      📊 Analyzing song (loading + 68 queries)... ✅ (18.23s)
      [1/68] Comparing: "piano" → 0.4118 ✅ HIGH (0.02s)
      ...

🎵 [2/3] Alter Ego - Rocker.mp3
      📊 Analyzing song (loading + 68 queries)... ✅ (19.45s)
      ...

🎵 [3/3] Another Song.mp3
      📊 Analyzing song (loading + 68 queries)... ✅ (20.11s)
      ...

================================================================================
🎵 🎵 Song Similarities
================================================================================

Top 3 most similar song pairs:

 1. 0.6234 ✅ Song1.mp3
                   ↔️  Song2.mp3
 2. 0.5123 ✅ Song2.mp3
                   ↔️  Song3.mp3
 3. 0.4567 ⚠️  Song1.mp3
                   ↔️  Song3.mp3

================================================================================
✅ Analysis complete!
💾 Results saved to: results_20251202_120000.csv
================================================================================
```

## Technical Details

**Embedding dimension:** 512 (CLAP audio encoder output)

**Normalization:** L2 norm (ensures dot product = cosine similarity)

**Similarity range:** [-1, 1] where:
- 1.0 = identical audio
- 0.0 = orthogonal (unrelated)
- -1.0 = opposite (rare in practice)

**Formula:**
```
similarity = dot(vec1, vec2) / (||vec1|| * ||vec2||)

# Since vectors are pre-normalized (||vec|| = 1):
similarity = dot(vec1, vec2)
```
