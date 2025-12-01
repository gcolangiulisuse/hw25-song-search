# Song Search (Hackweek 25)
<p align="center">
  <img src="https://github.com/gcolangiulisuse/hw25-song-search/blob/8b854433678c8f65317763a8e4bae35afa3d4a6c/HW25-SONG-SEARCH.png?raw=true" alt="SUSE Hackweek AI Song Search" width="600">
</p>

This repository hosts the code for my **openSUSE Hackweek 25** project.

The primary goal is to evaluate the **CLAP (Contrastive Language-Audio Pretraining)** library for searching songs using natural language. Future roadmap items include automated song tagging and LLM integration.

# Project Home & Updates
For full project details please visit the official Hackweek page:

👉 **[Hackweek 25 Project: Song Search with CLAP](https://hackweek.opensuse.org/25/projects/clap-machine-learning-to-search-song-starting-from-text)**

# How to Execute

```bash
# 1. Create virtual environment
python3 -m venv venv

# 2. Activate virtual environment
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run analysis
python clap_analysis.py ./songs/HoliznaCC0\ -\ Dreams\ Of\ Lilith\ -\ Rock.mp3 "Electric guitar songs"
python clap_analysis.py ./songs/Zane\ Little\ -\ Always\ and\ Forever\ -\ Pop.mp3 "Electrict guitar songs"
python clap_analysis.py ./songs/Zane\ Little\ -\ Always\ and\ Forever\ -\ Pop.mp3 "Pop relax songs"

# 5. Deactivate when done
deactivate
```

# Sample test

**Expected result: good match** 
```
python clap_analysis.py ./songs/HoliznaCC0\ -\ Dreams\ Of\ Lilith\ -\ Rock.mp3 "Electric guitar songs"
python clap_analysis.py ./songs/Zane\ Little\ -\ Always\ and\ Forever\ -\ Pop.mp3 "Pop relax songs"
```


**Expected result: bad match** 
```
python clap_analysis.py ./songs/Zane\ Little\ -\ Always\ and\ Forever\ -\ Pop.mp3 "Electrict guitar songs"
```