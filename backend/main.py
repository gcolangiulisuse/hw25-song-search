"""
Simplified Song Search Web API - No WebSocket, Simple Polling
Based on original clap_analysis.py approach
"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict
import multiprocessing
import numpy as np  # Added for vectorized operations

# CRITICAL: Force fork mode for multiprocessing to share model memory
multiprocessing.set_start_method('fork', force=True)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from database import Database
from analysis import AudioAnalyzer, find_audio_files


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="Song Search API")

# Global instances
db = Database("/app/config/analysis.db")
analyzer = AudioAnalyzer()

# ============================================================================
# IN-MEMORY CACHE (RAM)
# ============================================================================
# Storing data in RAM prevents reading from SQLite disk for every search
CACHED_SONGS: List[Dict] = []
CACHED_AUDIO_MATRIX: Optional[np.ndarray] = None

def refresh_memory_cache():
    """Load all songs and embeddings from DB into RAM."""
    global CACHED_SONGS, CACHED_AUDIO_MATRIX
    try:
        print("🧠 Loading database into RAM...")
        songs = db.get_all_songs()
        if songs:
            CACHED_SONGS = songs
            # Create the matrix once so we don't do it on every request
            CACHED_AUDIO_MATRIX = np.stack([s['embedding'] for s in songs])
            print(f"✅ Cached {len(songs)} songs in RAM")
        else:
            CACHED_SONGS = []
            CACHED_AUDIO_MATRIX = None
            print("⚠️ No songs to load into RAM")
    except Exception as e:
        print(f"❌ Failed to cache songs: {e}")


# ============================================================================
# Request Models
# ============================================================================

class ConfigUpdate(BaseModel):
    songs_path: str

class TextSearchRequest(BaseModel):
    query: str
    limit: int = 20

class SimilaritySearchRequest(BaseModel):
    song_id: int
    limit: int = 20


# ============================================================================
# Startup
# ============================================================================

@app.on_event("startup")
async def startup():
    """Print startup info and auto-configure songs path."""
    print("\n" + "="*80)
    print("🎵 Song Search Web App Starting...")
    print("="*80)
    
    # 1. Pre-initialize Model (Avoids delay on first search)
    print("🤖 Pre-initializing AI Model (this may take a moment)...")
    try:
        analyzer.initialize_model()
    except Exception as e:
        print(f"❌ Model initialization warning: {e}")

    # 2. Pre-load DB into RAM
    refresh_memory_cache()

    print()
    
    # Reset any stuck analysis state from previous runs
    progress = db.get_progress()
    if progress and progress.get('is_running'):
        print("🔄 Resetting stuck analysis state from previous run...")
        db.update_progress(
            is_running=False,
            current_song="",
            progress_pct=0,
            status_message="Application restarted"
        )
    
    # Auto-configure songs path from Docker mount
    default_songs_path = "/app/data/songs"
    current_path = db.get_songs_path()
    
    if not current_path and os.path.exists(default_songs_path):
        print(f"📁 Auto-configuring songs path: {default_songs_path}")
        db.set_songs_path(default_songs_path)
        audio_files = find_audio_files(default_songs_path)
        print(f"   Found {len(audio_files)} audio files")
    
    print("📊 Database Statistics:")
    stats = db.get_stats()
    print(f"   Songs analyzed: {stats['total_songs']}")
    print(f"   Unique queries: {stats['total_queries']}")
    print(f"   Total duration: {stats['total_duration_hours']:.2f} hours")
    
    songs_path = db.get_songs_path()
    print(f"   Songs path: {songs_path}")
    
    print("\n✅ Ready! Access the web interface at http://localhost:8000")
    print("="*80 + "\n")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/api/config")
async def get_config():
    """Get current configuration."""
    songs_path = db.get_songs_path()
    stats = db.get_stats()
    progress = db.get_progress()
    
    return {
        'songs_path': songs_path,
        'stats': stats,
        'analysis_progress': progress
    }


@app.post("/api/config")
async def update_config(config: ConfigUpdate):
    """Update configuration."""
    if not config.songs_path or not config.songs_path.strip():
        raise HTTPException(status_code=400, detail="Songs path cannot be empty")
    
    if not os.path.exists(config.songs_path):
        raise HTTPException(status_code=400, detail=f"Path does not exist: {config.songs_path}")
    
    if not os.path.isdir(config.songs_path):
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {config.songs_path}")
    
    if not os.access(config.songs_path, os.R_OK):
        raise HTTPException(status_code=403, detail=f"No read permission for directory: {config.songs_path}")
    
    try:
        db.set_songs_path(config.songs_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save configuration: {str(e)}")
    
    try:
        audio_files = find_audio_files(config.songs_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scan directory: {str(e)}")
    
    if len(audio_files) == 0:
        raise HTTPException(
            status_code=400, 
            detail=f"No audio files found in {config.songs_path}. Supported formats: mp3, wav, flac, ogg, m4a"
        )
    
    return {
        'success': True,
        'songs_path': config.songs_path,
        'audio_files_found': len(audio_files)
    }


@app.post("/api/analysis/start")
async def start_analysis(background_tasks: BackgroundTasks):
    """Start background analysis of all songs."""
    progress = db.get_progress()
    if progress['is_running']:
        raise HTTPException(status_code=400, detail="Analysis already running. Please wait for it to complete.")
    
    songs_path = db.get_songs_path()
    if not songs_path:
        raise HTTPException(
            status_code=400, 
            detail="Songs path not configured. Please go to Config tab and set your songs directory first."
        )
    
    if not os.path.exists(songs_path):
        raise HTTPException(
            status_code=400, 
            detail=f"Configured songs path no longer exists: {songs_path}"
        )
    
    try:
        audio_files = find_audio_files(songs_path)
        if len(audio_files) == 0:
            raise HTTPException(
                status_code=400, 
                detail=f"No audio files found in {songs_path}"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scan directory: {str(e)}")
    
    # Start background task
    background_tasks.add_task(run_analysis, songs_path)
    
    return {'success': True, 'message': 'Analysis started', 'total_files': len(audio_files)}


@app.get("/api/analysis/status")
async def get_analysis_status():
    """Get current analysis status (for polling)."""
    return db.get_progress()


@app.post("/api/search/text")
async def search_by_text(request: TextSearchRequest):
    """Search songs by text query using Vectorized Optimization (O(1)) and RAM Cache."""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty")
    
    if len(request.query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Search query must be at least 3 characters long")
    
    # Check cache first
    try:
        cached_results = db.search_by_text(request.query.strip(), limit=request.limit)
        if cached_results:
            return {
                'query': request.query.strip(),
                'results': cached_results,
                'cached': True
            }
    except Exception as e:
        print(f"⚠️ DB Cache lookup failed, proceeding to compute: {e}")
    
    # 1. Access Data from RAM (Instant)
    global CACHED_SONGS, CACHED_AUDIO_MATRIX
    
    if CACHED_AUDIO_MATRIX is None or not CACHED_SONGS:
        # Try refreshing if empty (maybe analysis just finished)
        refresh_memory_cache()
        if CACHED_AUDIO_MATRIX is None:
             raise HTTPException(status_code=400, detail="No songs analyzed. Please run analysis first.")
    
    # 2. Initialize model if needed (Should be pre-loaded at startup)
    if analyzer.model is None:
        try:
            analyzer.initialize_model()
        except Exception as e:
            print(f"\n❌ Failed to initialize model: {e}\n")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to initialize model: {str(e)}")
    
    # 3. Vectorized Similarity Computation
    try:
        query_text = request.query.strip()
        
        # A. Get Text Embedding ONCE (Heavy Operation - ~0.05s)
        text_embedding = analyzer.get_text_embedding(query_text)
        
        # B. Matrix Multiplication (Instant - Using RAM Cache)
        # (N, 512) @ (512,) -> (N,)
        similarities = CACHED_AUDIO_MATRIX @ text_embedding
        
        # 4. Format Results
        all_results = []
        
        # We iterate to format the response and save to DB query cache.
        for i, song in enumerate(CACHED_SONGS):
            score = float(similarities[i])
            
            all_results.append({
                'id': song['id'],
                'filename': song['filename'],
                'filepath': song['filepath'],
                'artist': song['artist'],
                'title': song['title'],
                'duration_sec': song['duration_sec'],
                'similarity': score
            })
            
            # Cache individual result to DB for future exact-match lookups
            db.add_query_result(
                song_id=song['id'],
                query_text=query_text,
                similarity=score,
                max_score=score,
                min_score=score,
                std_score=0.0
            )

    except Exception as e:
        print(f"❌ Critical error in vectorized search: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to compute similarities: {str(e)}")
    
    if not all_results:
        raise HTTPException(status_code=500, detail="Failed to compute any similarities.")
    
    # Sort by similarity
    all_results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return {
        'query': request.query.strip(),
        'results': all_results[:request.limit],
        'cached': False
    }


@app.post("/api/search/similar")
async def search_similar(request: SimilaritySearchRequest):
    """Find similar songs using Vectorized Optimization (O(1)) and RAM Cache."""
    if request.song_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid song ID")
    
    # 1. Access Data from RAM
    global CACHED_SONGS, CACHED_AUDIO_MATRIX
    
    if CACHED_AUDIO_MATRIX is None or not CACHED_SONGS:
        refresh_memory_cache()
        if CACHED_AUDIO_MATRIX is None:
             raise HTTPException(status_code=400, detail="No songs analyzed. Please run analysis first.")

    # 2. Find target song in cache (Fast Linear Search in RAM)
    target_index = -1
    target_song = None
    
    for i, song in enumerate(CACHED_SONGS):
        if song['id'] == request.song_id:
            target_index = i
            target_song = song
            break
            
    if target_index == -1:
         raise HTTPException(status_code=404, detail=f"Song with ID {request.song_id} not found in cache.")

    # 3. Vectorized Similarity Computation
    try:
        # Get embedding directly from RAM Matrix
        target_embedding = CACHED_AUDIO_MATRIX[target_index]
        
        # Matrix Multiplication (Instant)
        # (N, 512) @ (512,) -> (N,)
        similarities = CACHED_AUDIO_MATRIX @ target_embedding
        
        # 4. Format Results
        scored_songs = []
        for i, song in enumerate(CACHED_SONGS):
            score = float(similarities[i])
            scored_songs.append((score, song))
            
        # Sort by score descending
        scored_songs.sort(key=lambda x: x[0], reverse=True)
        
        # Take limit
        results = []
        for score, song in scored_songs[:request.limit]:
             results.append({
                'id': song['id'],
                'filename': song['filename'],
                'filepath': song['filepath'],
                'artist': song['artist'],
                'title': song['title'],
                'duration_sec': song['duration_sec'],
                'similarity': score
            })
    
    except Exception as e:
        print(f"❌ Critical error in vectorized similar search: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to find similar songs: {str(e)}")
    
    return {
        'song_id': request.song_id,
        'target_song': {
            'artist': target_song['artist'],
            'title': target_song['title'],
            'filename': target_song['filename']
        },
        'results': results
    }


@app.get("/api/songs")
async def get_all_songs():
    """Get all analyzed songs."""
    songs = db.get_all_songs()
    
    results = []
    for song in songs:
        results.append({
            'id': song['id'],
            'filename': song['filename'],
            'artist': song['artist'],
            'title': song['title'],
            'duration_sec': song['duration_sec'],
            'analyzed_at': song['analyzed_at']
        })
    
    return {'songs': results}


@app.get("/api/audio/{song_id}")
async def stream_audio(song_id: int):
    """Stream audio file."""
    if song_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid song ID")
    
    try:
        song = db.get_song(song_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    if not song:
        raise HTTPException(status_code=404, detail=f"Song with ID {song_id} not found")
    
    filepath = song['filepath']
    
    if not os.path.exists(filepath):
        raise HTTPException(
            status_code=404, 
            detail=f"Audio file not found on disk: {filepath}. The file may have been moved or deleted."
        )
    
    if not os.access(filepath, os.R_OK):
        raise HTTPException(
            status_code=403, 
            detail=f"No read permission for audio file: {filepath}"
        )
    
    # Determine media type
    ext = os.path.splitext(filepath)[1].lower()
    media_types = {
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg',
        '.m4a': 'audio/mp4'
    }
    media_type = media_types.get(ext, 'audio/mpeg')
    
    try:
        return FileResponse(
            filepath,
            media_type=media_type,
            filename=song['filename']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stream audio: {str(e)}")


# ============================================================================
# Background Analysis Task (Optimized for Resumability)
# ============================================================================

async def run_analysis(songs_path: str):
    """Background task to analyze only new songs sequentially."""
    db.update_progress(True, '', 0, 0, 0.0, 'Starting analysis...')
    
    try:
        # Clear query cache so old songs aren't excluded from new searches
        db.clear_query_cache()
        
        # Initialize model
        db.update_progress(True, '', 0, 0, 0.0, 'Initializing CLAP model...')
        
        if analyzer.model is None:
            try:
                analyzer.initialize_model()
            except Exception as e:
                error_msg = f'Failed to initialize model: {str(e)}'
                print(f"\n❌ {error_msg}\n")
                import traceback
                traceback.print_exc()
                db.update_progress(False, '', 0, 0, 0.0, error_msg)
                return
        
        # 1. Find all audio files on disk
        print(f"📂 Scanning directory: {songs_path}")
        audio_files = find_audio_files(songs_path)
        total_files = len(audio_files)
        
        if total_files == 0:
            db.update_progress(False, '', 0, 0, 0.0, 'No audio files found')
            print("⚠️  No audio files found in directory")
            return
            
        # 2. Get list of already analyzed songs (using efficient Set lookup)
        # This prevents the loop from having to query the DB for every single skipped file
        print("📊 Fetching existing database records...")
        existing_filenames = db.get_all_filenames()
        already_analyzed_count = 0
        
        # 3. Filter list to process ONLY new files
        files_to_process = []
        for f in audio_files:
            if os.path.basename(f) in existing_filenames:
                already_analyzed_count += 1
            else:
                files_to_process.append(f)
        
        to_analyze_count = len(files_to_process)
        
        # Immediate success if nothing to do
        if to_analyze_count == 0:
            msg = f'All {total_files} songs already analyzed. No new songs to process.'
            db.update_progress(False, '', total_files, total_files, 100.0, msg)
            print(f"\n✅ {msg}\n")
            return

        print(f"\n📊 Analysis Plan:")
        print(f"   Total found: {total_files}")
        print(f"   Skipping:    {already_analyzed_count} (already in DB)")
        print(f"   To Analyze:  {to_analyze_count}")
        
        # Update progress to show we jumped ahead
        current_progress_pct = (already_analyzed_count / total_files) * 100
        db.update_progress(True, '', already_analyzed_count, total_files, current_progress_pct, 
                          f'Skipped {already_analyzed_count} files, starting analysis of {to_analyze_count} new files...')
        
        # 4. Sequential Processing Loop
        # We use a simple FOR loop + await to guarantee STRICT sequential processing.
        # This fixes the "starting multiple" crash issue completely.
        
        for i, audio_path in enumerate(files_to_process):
            # Calculate global index (including skipped ones)
            global_idx = already_analyzed_count + i + 1
            filename = os.path.basename(audio_path)
            
            print(f"\n📝 Processing {global_idx}/{total_files}: {filename}")
            
            progress_pct = (global_idx / total_files) * 100
            db.update_progress(True, filename, global_idx, total_files, progress_pct, 
                             f'Analyzing {i+1}/{to_analyze_count}: {filename}...')
            
            try:
                print(f"🎵 Starting analysis of {filename}...")
                
                # Analyze song - runs in executor to avoid blocking
                loop = asyncio.get_event_loop()
                try:
                    embedding, duration_sec, num_segments = await asyncio.wait_for(
                        loop.run_in_executor(None, analyzer.analyze_song, audio_path),
                        timeout=180  # 3 minutes max per song
                    )
                except asyncio.TimeoutError:
                    raise RuntimeError(f"Analysis timed out after 3 minutes")
                
                print(f"✅ Analyzed {filename}: {num_segments} segments, {duration_sec:.1f}s")
                
                # Get metadata
                artist, title = await loop.run_in_executor(
                    None, analyzer.get_metadata, audio_path
                )
                
                # Get file stats
                file_stats = os.stat(audio_path)
                file_size = file_stats.st_size
                file_modified = datetime.fromtimestamp(file_stats.st_mtime)
                
                # Save to database
                db.add_song(
                    filename=filename,
                    filepath=audio_path,
                    duration_sec=duration_sec,
                    num_segments=num_segments,
                    embedding=embedding,
                    artist=artist,
                    title=title,
                    file_size_bytes=file_size,
                    file_modified_at=file_modified
                )
                
                print(f"💾 Saved {filename} to database")
            
            except Exception as e:
                error_msg = str(e)
                print(f"\n❌ Error analyzing {filename}: {error_msg}")
                
                # Safely reset pool if needed
                print("⚠️  Resetting thread pool after error...")
                try:
                    # Fix: use the correct method from analysis.py
                    if hasattr(analyzer, 'close_pool'):
                        analyzer.close_pool()
                    elif hasattr(analyzer, 'executor') and analyzer.executor:
                        analyzer.executor.shutdown(wait=False)
                        analyzer.executor = None
                except Exception as pool_err:
                    print(f"⚠️  Error closing pool: {pool_err}")
                
                print(f"⏭️  Skipping {filename} and continuing with next song...\n")
                # Don't crash the whole loop, just this song
                import traceback
                traceback.print_exc()

        # Complete
        db.clear_query_cache()
        
        # REFRESH MEMORY CACHE
        refresh_memory_cache()
        
        final_analyzed = db.get_songs_count()
        msg = f'Complete! {final_analyzed} songs in database ({to_analyze_count} newly analyzed).'
        db.update_progress(False, '', total_files, total_files, 100.0, msg)
        print(f"\n🎉 Analysis complete! {final_analyzed} songs in database.\n")
    
    except Exception as e:
        error_msg = f'Fatal Analysis Error: {str(e)}'
        db.update_progress(False, '', 0, 0, 0.0, error_msg)
        print(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# Frontend
# ============================================================================

@app.get("/")
async def index():
    """Serve frontend."""
    frontend_path = Path(__file__).parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    else:
        return HTMLResponse("<h1>Frontend not found. Please build the frontend.</h1>")


# Mount static files
frontend_static = Path(__file__).parent.parent / "frontend"
if frontend_static.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_static)), name="static")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )