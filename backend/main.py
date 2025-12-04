"""
Simplified Song Search Web API - No WebSocket, Simple Polling
Based on original clap_analysis.py approach
"""

import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import multiprocessing

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
    print("   Model will be initialized on first analysis or search")
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
    """Search songs by text query."""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty")
    
    if len(request.query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Search query must be at least 3 characters long")
    
    songs_count = db.get_songs_count()
    if songs_count == 0:
        raise HTTPException(
            status_code=400, 
            detail="No songs analyzed yet. Please go to the Analysis tab and analyze your songs first."
        )
    
    # Check cache first
    try:
        results = db.search_by_text(request.query.strip(), limit=request.limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    if results:
        return {
            'query': request.query.strip(),
            'results': results,
            'cached': True
        }
    
    # Compute similarities
    try:
        songs = db.get_all_songs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load songs: {str(e)}")
    
    if not songs:
        raise HTTPException(status_code=400, detail="No songs available. Please run analysis first.")
    
    # Initialize model if needed
    if analyzer.model is None:
        try:
            analyzer.initialize_model()
        except Exception as e:
            print(f"\n❌ Failed to initialize model: {e}\n")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to initialize model: {str(e)}")
    
    # Compute similarities for all songs using parallel threads
    all_results = []
    try:
        # Prepare data for parallel processing
        query_text = request.query.strip()
        
        # Use ThreadPoolExecutor for parallel text similarity computation
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os
        
        def compute_similarity_for_song(song):
            """Worker function to compute similarity for one song."""
            try:
                similarity = analyzer.compute_text_similarity(song['embedding'], query_text)
                return {
                    'id': song['id'],
                    'filename': song['filename'],
                    'filepath': song['filepath'],
                    'artist': song['artist'],
                    'title': song['title'],
                    'duration_sec': song['duration_sec'],
                    'similarity': similarity
                }
            except Exception as e:
                print(f"❌ Failed to compute similarity for {song['filename']}: {e}")
                return None
        
        # Process in parallel with threads (no pickling issues)
        max_workers = min(os.cpu_count() or 4, len(songs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(compute_similarity_for_song, song): song for song in songs}
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_results.append(result)
                    
                    # Cache result
                    db.add_query_result(
                        song_id=result['id'],
                        query_text=query_text,
                        similarity=result['similarity'],
                        max_score=result['similarity'],
                        min_score=result['similarity'],
                        std_score=0.0
                    )
    except Exception as e:
        print(f"❌ Critical error in similarity computation: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to compute similarities: {str(e)}")
    
    if not all_results:
        raise HTTPException(status_code=500, detail="Failed to compute any similarities. Please try again.")
    
    # Sort by similarity
    all_results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return {
        'query': request.query.strip(),
        'results': all_results[:request.limit],
        'cached': False
    }


@app.post("/api/search/similar")
async def search_similar(request: SimilaritySearchRequest):
    """Find similar songs."""
    if request.song_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid song ID")
    
    try:
        target_song = db.get_song(request.song_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    if not target_song:
        raise HTTPException(status_code=404, detail=f"Song with ID {request.song_id} not found")
    
    try:
        similar_songs = db.get_similar_songs(request.song_id, limit=request.limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to find similar songs: {str(e)}")
    
    results = []
    for song, similarity in similar_songs:
        results.append({
            'id': song['id'],
            'filename': song['filename'],
            'filepath': song['filepath'],
            'artist': song['artist'],
            'title': song['title'],
            'duration_sec': song['duration_sec'],
            'similarity': similarity
        })
    
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
# Background Analysis Task (Like Original clap_analysis.py)
# ============================================================================

async def run_analysis(songs_path: str):
    """Background task to analyze all songs - ORIGINAL APPROACH."""
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
        
        # Find audio files
        audio_files = find_audio_files(songs_path)
        total = len(audio_files)
        
        if total == 0:
            db.update_progress(False, '', 0, 0, 0.0, 'No audio files found')
            print("⚠️  No audio files found in directory")
            return
        
        # Count already analyzed
        already_analyzed = sum(1 for f in audio_files if db.song_exists(os.path.basename(f)))
        to_analyze = total - already_analyzed
        
        if to_analyze == 0:
            msg = f'All {total} songs already analyzed. No new songs to process.'
            db.update_progress(False, '', total, total, 100.0, msg)
            print(f"\n✅ {msg}\n")
            return
        
        print(f"\n📊 Found {total} songs: {already_analyzed} already analyzed, {to_analyze} new\n")
        
        # Analyze each song (ORIGINAL METHOD - uses multiprocessing internally)
        for idx, audio_path in enumerate(audio_files, 1):
            filename = os.path.basename(audio_path)
            
            print(f"\n📝 Processing {idx}/{total}: {filename}")
            
            progress_pct = (idx / total) * 100
            db.update_progress(True, filename, idx, total, progress_pct, f'Processing {idx}/{total}: {filename}...')
            
            try:
                # Check if already analyzed
                if db.song_exists(filename):
                    print(f"⏭️  Skipping {filename} (already analyzed)")
                    db.update_progress(True, filename, idx, total, progress_pct, f'Skipping {filename} (already analyzed)')
                    await asyncio.sleep(0.1)
                    continue
                
                print(f"🎵 Starting analysis of {filename}...")
                
                # Analyze song - runs in executor to avoid blocking
                # This uses multiprocessing internally (original clap_analysis.py approach)
                # Add asyncio timeout as additional safety measure
                loop = asyncio.get_event_loop()
                try:
                    embedding, duration_sec, num_segments = await asyncio.wait_for(
                        loop.run_in_executor(None, analyzer.analyze_song, audio_path),
                        timeout=120  # 2 minutes max per song
                    )
                except asyncio.TimeoutError:
                    raise RuntimeError(f"Analysis timed out after 2 minutes")
                
                print(f"✅ Analyzed {filename}: {num_segments} segments, {duration_sec:.1f}s")
                
                # Get metadata
                artist, title = await loop.run_in_executor(
                    None, analyzer.get_metadata, audio_path
                )
                
                print(f"📋 Metadata: {artist} - {title}")
                
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
                
                db.update_progress(True, filename, idx, total, progress_pct, f'Completed {filename}')
            
            except Exception as e:
                error_msg = str(e)
                print(f"\n❌ Error analyzing {filename}: {error_msg}")
                
                # Terminate corrupted pool after timeout/error
                if analyzer.pool:
                    print("⚠️  Terminating corrupted pool after error...")
                    analyzer.pool.terminate()
                    analyzer.pool.join()
                    analyzer.pool = None
                    print("✅ Pool terminated, will be recreated for next song")
                
                print(f"⏭️  Skipping {filename} and continuing with next song...\n")
                import traceback
                traceback.print_exc()
                db.update_progress(True, filename, idx, total, progress_pct, f'Error with {filename} (skipped): {error_msg}')
                # Continue with next song instead of stopping
                await asyncio.sleep(0.1)
                continue
        
        # Complete - Clear query cache again so new searches include all songs
        db.clear_query_cache()
        
        final_analyzed = db.get_songs_count()
        msg = f'Complete! {final_analyzed} songs in database ({to_analyze} newly analyzed, {already_analyzed} already existed).'
        db.update_progress(False, '', total, total, 100.0, msg)
        print(f"\n🎉 Analysis complete! {final_analyzed} songs in database ({to_analyze} newly analyzed).\n")
    
    except Exception as e:
        error_msg = f'Error: {str(e)}'
        db.update_progress(False, '', 0, 0, 0.0, error_msg)
        print(f"Analysis error: {e}")


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
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
