"""
Database operations for song analysis storage.
Uses SQLite for lightweight, embedded storage.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np


class Database:
    """SQLite database manager for song analysis."""
    
    def __init__(self, db_path: str = "/app/config/analysis.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Check if DB already exists
        db_exists = os.path.exists(db_path)
        
        if db_exists:
            print(f"✅ Loading existing database from: {db_path}")
        else:
            print(f"🆕 Creating new database at: {db_path}")
        
        # Connect to database
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Initialize schema if new database
        if not db_exists:
            self._init_schema()
    
    def _init_schema(self):
        """Create database schema."""
        cursor = self.conn.cursor()
        
        # Config table - stores app configuration
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Songs table - stores analyzed songs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS songs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                filepath TEXT NOT NULL,
                artist TEXT,
                title TEXT,
                duration_sec REAL NOT NULL,
                num_segments INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_size_bytes INTEGER,
                file_modified_at TIMESTAMP
            )
        """)
        
        # Query results table - stores text query analysis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                song_id INTEGER NOT NULL,
                query_text TEXT NOT NULL,
                similarity REAL NOT NULL,
                max_score REAL,
                min_score REAL,
                std_score REAL,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (song_id) REFERENCES songs(id) ON DELETE CASCADE,
                UNIQUE(song_id, query_text)
            )
        """)
        
        # Analysis progress table - for polling status
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_progress (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                is_running INTEGER DEFAULT 0,
                current_song TEXT DEFAULT '',
                current_idx INTEGER DEFAULT 0,
                total_songs INTEGER DEFAULT 0,
                progress_pct REAL DEFAULT 0.0,
                status_message TEXT DEFAULT '',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Initialize progress row
        cursor.execute("INSERT OR IGNORE INTO analysis_progress (id) VALUES (1)")
        
        # Indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_songs_filename ON songs(filename)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_query_results_song ON query_results(song_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_query_results_similarity ON query_results(similarity DESC)")
        
        self.conn.commit()
        print("✅ Database schema initialized")
    
    def close(self):
        """Close database connection."""
        self.conn.close()
    
    # ========================================================================
    # Config operations
    # ========================================================================
    
    def get_config(self, key: str) -> Optional[str]:
        """Get configuration value."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM config WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else None
    
    def set_config(self, key: str, value: str):
        """Set configuration value."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO config (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = CURRENT_TIMESTAMP
        """, (key, value, value))
        self.conn.commit()
    
    def get_songs_path(self) -> Optional[str]:
        """Get configured songs directory path."""
        return self.get_config('songs_path')
    
    def set_songs_path(self, path: str):
        """Set songs directory path."""
        self.set_config('songs_path', path)
    
    # ========================================================================
    # Song operations
    # ========================================================================
    
    def add_song(self, filename: str, filepath: str, duration_sec: float, 
                 num_segments: int, embedding: np.ndarray, 
                 artist: Optional[str] = None, title: Optional[str] = None,
                 file_size_bytes: Optional[int] = None,
                 file_modified_at: Optional[datetime] = None) -> int:
        """
        Add or update a song in the database.
        
        Returns:
            song_id: ID of the inserted/updated song
        """
        cursor = self.conn.cursor()
        
        # Convert embedding to bytes
        embedding_bytes = embedding.tobytes()
        
        cursor.execute("""
            INSERT INTO songs (filename, filepath, artist, title, duration_sec, 
                             num_segments, embedding, file_size_bytes, file_modified_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(filename) DO UPDATE SET
                filepath = excluded.filepath,
                artist = excluded.artist,
                title = excluded.title,
                duration_sec = excluded.duration_sec,
                num_segments = excluded.num_segments,
                embedding = excluded.embedding,
                analyzed_at = CURRENT_TIMESTAMP,
                file_size_bytes = excluded.file_size_bytes,
                file_modified_at = excluded.file_modified_at
        """, (filename, filepath, artist, title, duration_sec, num_segments, 
              embedding_bytes, file_size_bytes, file_modified_at))
        
        self.conn.commit()
        
        # Get the song ID
        cursor.execute("SELECT id FROM songs WHERE filename = ?", (filename,))
        return cursor.fetchone()[0]
    
    def get_song(self, song_id: int) -> Optional[Dict]:
        """Get song by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM songs WHERE id = ?", (song_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return {
            'id': row['id'],
            'filename': row['filename'],
            'filepath': row['filepath'],
            'artist': row['artist'],
            'title': row['title'],
            'duration_sec': row['duration_sec'],
            'num_segments': row['num_segments'],
            'embedding': np.frombuffer(row['embedding'], dtype=np.float32),
            'analyzed_at': row['analyzed_at'],
            'file_size_bytes': row['file_size_bytes'],
            'file_modified_at': row['file_modified_at']
        }
    
    def get_song_by_filename(self, filename: str) -> Optional[Dict]:
        """Get song by filename."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM songs WHERE filename = ?", (filename,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return {
            'id': row['id'],
            'filename': row['filename'],
            'filepath': row['filepath'],
            'artist': row['artist'],
            'title': row['title'],
            'duration_sec': row['duration_sec'],
            'num_segments': row['num_segments'],
            'embedding': np.frombuffer(row['embedding'], dtype=np.float32),
            'analyzed_at': row['analyzed_at'],
            'file_size_bytes': row['file_size_bytes'],
            'file_modified_at': row['file_modified_at']
        }
    
    def get_all_songs(self) -> List[Dict]:
        """Get all songs."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM songs ORDER BY filename")
        rows = cursor.fetchall()
        
        songs = []
        for row in rows:
            songs.append({
                'id': row['id'],
                'filename': row['filename'],
                'filepath': row['filepath'],
                'artist': row['artist'],
                'title': row['title'],
                'duration_sec': row['duration_sec'],
                'num_segments': row['num_segments'],
                'embedding': np.frombuffer(row['embedding'], dtype=np.float32),
                'analyzed_at': row['analyzed_at'],
                'file_size_bytes': row['file_size_bytes'],
                'file_modified_at': row['file_modified_at']
            })
        
        return songs
    
    def get_songs_count(self) -> int:
        """Get total number of analyzed songs."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM songs")
        return cursor.fetchone()[0]
    
    def song_exists(self, filename: str) -> bool:
        """Check if song exists in database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM songs WHERE filename = ? LIMIT 1", (filename,))
        return cursor.fetchone() is not None
    
    def delete_song(self, song_id: int):
        """Delete a song (cascade deletes query results too)."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM songs WHERE id = ?", (song_id,))
        self.conn.commit()
    
    # ========================================================================
    # Query results operations
    # ========================================================================
    
    def clear_query_cache(self):
        """Clear all cached query results. Call this when starting new analysis."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM query_results")
        self.conn.commit()
        print("🗑️  Cleared query cache")
    
    def add_query_result(self, song_id: int, query_text: str, similarity: float,
                        max_score: float, min_score: float, std_score: float):
        """Add or update a query result."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO query_results (song_id, query_text, similarity, max_score, min_score, std_score)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(song_id, query_text) DO UPDATE SET
                similarity = excluded.similarity,
                max_score = excluded.max_score,
                min_score = excluded.min_score,
                std_score = excluded.std_score,
                analyzed_at = CURRENT_TIMESTAMP
        """, (song_id, query_text, similarity, max_score, min_score, std_score))
        self.conn.commit()
    
    def search_by_text(self, query_text: str, limit: int = 20) -> List[Dict]:
        """
        Search songs by text query.
        Returns top matches ordered by similarity.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT s.id, s.filename, s.filepath, s.artist, s.title, s.duration_sec,
                   qr.similarity, qr.query_text
            FROM query_results qr
            JOIN songs s ON s.id = qr.song_id
            WHERE qr.query_text = ?
            ORDER BY qr.similarity DESC
            LIMIT ?
        """, (query_text, limit))
        
        rows = cursor.fetchall()
        results = []
        for row in rows:
            results.append({
                'id': row['id'],
                'filename': row['filename'],
                'filepath': row['filepath'],
                'artist': row['artist'],
                'title': row['title'],
                'duration_sec': row['duration_sec'],
                'similarity': row['similarity'],
                'query_text': row['query_text']
            })
        
        return results
    
    def get_similar_songs(self, song_id: int, limit: int = 20) -> List[Tuple[Dict, float]]:
        """
        Find songs similar to the given song.
        Computes cosine similarity using stored embeddings.
        
        Returns:
            List of (song_dict, similarity_score) tuples
        """
        # Get the target song's embedding
        target_song = self.get_song(song_id)
        if not target_song:
            return []
        
        target_embedding = target_song['embedding']
        
        # Get all other songs
        all_songs = self.get_all_songs()
        
        # Compute similarities
        similarities = []
        for song in all_songs:
            if song['id'] == song_id:
                continue
            
            # Cosine similarity (embeddings are already normalized)
            similarity = float(np.dot(target_embedding, song['embedding']))
            
            similarities.append((song, similarity))
        
        # Sort by similarity (highest first) and limit
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Add the searched song itself at position 0 with similarity 1.0
        results = [(target_song, 1.0)] + similarities[:limit-1]
        
        return results
    
    # ========================================================================
    # Statistics
    # ========================================================================
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM songs")
        total_songs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT query_text) FROM query_results")
        total_queries = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(duration_sec) FROM songs")
        total_duration = cursor.fetchone()[0] or 0
        
        return {
            'total_songs': total_songs,
            'total_queries': total_queries,
            'total_duration_sec': total_duration,
            'total_duration_hours': total_duration / 3600
        }
    
    # ========================================================================
    # Analysis Progress (for polling)
    # ========================================================================
    
    def update_progress(self, is_running: bool, current_song: str = '', 
                       current_idx: int = 0, total_songs: int = 0, 
                       progress_pct: float = 0.0, status_message: str = ''):
        """Update analysis progress in database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE analysis_progress 
            SET is_running = ?, current_song = ?, current_idx = ?, 
                total_songs = ?, progress_pct = ?, status_message = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
        """, (1 if is_running else 0, current_song, current_idx, 
              total_songs, progress_pct, status_message))
        self.conn.commit()
    
    def get_progress(self) -> Dict:
        """Get current analysis progress from database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM analysis_progress WHERE id = 1")
        row = cursor.fetchone()
        
        if row:
            return {
                'is_running': bool(row['is_running']),
                'current_song': row['current_song'] or '',
                'current_idx': row['current_idx'] or 0,
                'total_songs': row['total_songs'] or 0,
                'progress_pct': row['progress_pct'] or 0.0,
                'status_message': row['status_message'] or ''
            }
        return {
            'is_running': False,
            'current_song': '',
            'current_idx': 0,
            'total_songs': 0,
            'progress_pct': 0.0,
            'status_message': ''
        }
