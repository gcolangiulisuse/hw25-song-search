# Query Wording Sensitivity Analysis

**Date:** 2 December 2025

## Problem: Word Choice Matters

CLAP's text encoder is **highly sensitive to exact wording**. Synonyms and similar phrases can produce dramatically different similarity scores, even when describing the same musical characteristic.

## Real-World Examples

### Example 1: "vocalist" vs "voice"

**Song:** Paint it Green - SUSE Band.mp3 (female-sung rock song)

| Query | Score | Label | Δ from "vocalist" |
|-------|-------|-------|-------------------|
| `female vocalist` | **0.3283** | HIGH | baseline |
| `female voice` | **0.0616** | LOW | **-0.267** (-81%) |
| `male vocalist` | **0.1901** | MODERATE | -0.138 (-42%) |
| `male voice` | **-0.0328** | LOW | -0.361 (-110%) |

**Analysis:** The word "**vocalist**" produces 5× higher scores than "**voice**" for the same concept. The model was trained with "vocalist" as a more common descriptor in music contexts.

### Example 2: "pop songs with female vocalist" compound query

**Song:** Jenna Jay - Someone Real (female pop song)

| Query | Score | Label | Interpretation |
|-------|-------|-------|----------------|
| `female vocalist` | **0.4420** | HIGH | Strong match |
| `pop songs with female vocalist` | **0.4116** | HIGH | Slightly lower |
| `female voice` | **0.1330** | LOW | Weak match |

**Analysis:** Compound queries work but slightly dilute the score. The model averages across all words in the query.

### Example 3: Instrument naming

**Song:** Paint it Green - SUSE Band.mp3 (rock with guitar)

| Query | Score | Label |
|-------|-------|-------|
| `electric guitar` | **0.1102** | LOW |
| `guitar` | **(varies)** | - |
| `rock music` | **0.3180** | HIGH |

**Analysis:** Generic genre terms ("rock music") often score higher than specific instrument names, especially if the instrument isn't dominant in the mix.

### Example 4: Genre vs Instrument

**Song:** Aaron Dunn - Minuet (classical piano)

| Query | Score | Label |
|-------|-------|-------|
| `piano` | **0.4118** | HIGH |
| `classical music` | **0.3756** | HIGH |
| `classical instrumental song with piano` | **0.3958** | HIGH |

**Analysis:** Both instrument and genre work well when they're prominent. Compound queries maintain good scores.

## Key Findings

### 1. Training Data Vocabulary Matters
The model learned from millions of audio-text pairs. Common music terminology (e.g., "vocalist", "rock music", "piano") produces more reliable results than colloquial terms (e.g., "voice", "guitar player").

### 2. Specificity vs Generality Trade-off
- **Too specific:** "acoustic steel-string guitar" → may not match training vocabulary
- **Too general:** "sound" → too vague
- **Sweet spot:** "acoustic guitar" → common enough, specific enough

### 3. Compound Queries Average Scores
```
"pop music" (0.40) + "female vocalist" (0.35) 
→ "pop music with female vocalist" (0.37)
```
Each word/phrase contributes to the final embedding. More words = more averaging.

### 4. Negative Scores Are Normal
Scores range from -1 to +1. Negative scores indicate **opposite characteristics**, not errors:
- `male voice` on female-sung track → -0.033
- `rock music` on classical piano → -0.022

## Recommendations

### For Best Results

1. **Use Music Industry Terms**
   - ✅ "vocalist", "instrumental", "melody"
   - ❌ "singer", "no words", "tune"

2. **Keep Queries Simple**
   - ✅ "female vocalist"
   - ⚠️ "pop songs with female vocalist" (acceptable but diluted)
   - ❌ "upbeat pop songs with a female singer and synthesizers" (too complex)

3. **Test Synonyms**
   ```python
   queries = [
       "piano",
       "piano music",
       "instrumental piano",
       "classical piano"
   ]
   ```
   Run all variants to find which wording the model responds to best.

4. **Use Genre Terms for Whole-Song Characteristics**
   - ✅ "rock music", "classical music", "electronic music"
   - Less effective: listing every instrument

### Query Templates That Work Well

**Instruments:**
- `piano`
- `electric guitar`
- `acoustic guitar`
- `bass guitar`

**Vocals:**
- `female vocalist` (not "female voice")
- `male vocalist` (not "male voice")
- `choir`
- `singing`

**Genres:**
- `rock music`
- `classical music`
- `pop music`
- `electronic music`

**Combined:**
- `[genre] with [instrument]` → e.g., "jazz with saxophone"
- `[instrument] [genre]` → e.g., "piano classical"

## Technical Explanation

CLAP's text encoder (RoBERTa) tokenizes and embeds text into 512-dimensional vectors. Similar **word patterns from training data** create similar embeddings:

- Training had many examples: `"rock music with electric guitar"` ✓
- Training had fewer: `"rock sound with guitar voice"` ✗

The embedding space learned **music domain conventions**, not general language semantics. Always query using the vocabulary the model was trained on.

## Practical Example: Building a Query List

**Goal:** Find songs with female vocals

**Test these variations:**
```
female vocalist          # BEST - 0.3283 on test song
female singer           # Try this - may vary
female voice            # WORST - 0.0616 on test song
singing by a woman      # Avoid - too verbose
female vocals           # Try this - may work
```

**Result:** Use `"female vocalist"` as the primary query, and `"female vocals"` as a secondary check.

## Conclusion

Query wording dramatically affects results. The difference between "female vocalist" (HIGH) and "female voice" (LOW) can be **5× in similarity score**. 

**Always test multiple phrasings** of the same concept to find which vocabulary the model responds to best.
