// Song Search Web App - Alpine.js Application

function app() {
    return {
        // State
        currentTab: 'analysis',
        lastRefreshTab: '',
        pollInterval: null,
        analysisCompleted: false,
        stats: {
            total_songs: 0,
            total_queries: 0,
            total_duration_sec: 0
        },
        
        // Config
        songsPath: '',
        configMessage: '',
        configSuccess: false,
        
        // Analysis
        analysisState: {
            is_running: false,
            current_song: '',
            current_idx: 0,
            total_songs: 0,
            progress_pct: 0.0,
            status_message: ''
        },
        ws: null,
        
        // Text Search
        textQuery: '',
        textResults: [],
        searching: false,
        
        // Similarity Search
        selectedSongId: '',
        similarResults: [],
        allSongs: [],
        songSearchQuery: '',
        filteredSongs: [],
        
        // Audio Player
        currentlyPlaying: null,
        currentTime: 0,
        audioDuration: 0,
        
        // Initialize
        async init() {
            await this.loadConfig();
            await this.loadAllSongs();
            // Only start polling if analysis is running
            await this.checkAnalysisStatus();
        },
        
        // Tab switching - refresh data only when entering text-search or similarity tab
        async switchTab(tab) {
            this.currentTab = tab;
            
            // Refresh songs list when switching to search tabs (if not refreshed recently)
            if ((tab === 'text-search' || tab === 'similarity') && this.lastRefreshTab !== tab) {
                await this.loadConfig();
                await this.loadAllSongs();
                this.lastRefreshTab = tab;
            }
        },
        
        // Config Methods
        async loadConfig() {
            try {
                const response = await fetch('/api/config');
                const data = await response.json();
                
                this.songsPath = data.songs_path || '';
                this.stats = data.stats;
            } catch (error) {
                console.error('Error loading config:', error);
            }
        },
        
        async updateConfig() {
            try {
                const response = await fetch('/api/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ songs_path: this.songsPath })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    this.configMessage = `✅ Configuration saved! Found ${data.audio_files_found} audio files.`;
                    this.configSuccess = true;
                    await this.loadConfig();
                } else {
                    this.configMessage = '❌ Error saving configuration';
                    this.configSuccess = false;
                }
            } catch (error) {
                this.configMessage = `❌ Error: ${error.message}`;
                this.configSuccess = false;
            }
            
            // Clear message after 5 seconds
            setTimeout(() => {
                this.configMessage = '';
            }, 5000);
        },
        
        // Analysis Methods
        async startAnalysis() {
            // Immediately update UI to prevent multiple clicks
            this.analysisState.is_running = true;
            this.analysisState.status_message = 'Starting analysis...';
            
            try {
                const response = await fetch('/api/analysis/start', {
                    method: 'POST'
                });
                
                const data = await response.json();
                
                if (data.success) {
                    // Start checking status immediately
                    this.startPolling();
                } else {
                    // Reset state if request failed
                    this.analysisState.is_running = false;
                    this.analysisState.status_message = '';
                }
            } catch (error) {
                // Reset state on error
                this.analysisState.is_running = false;
                this.analysisState.status_message = '';
                alert(`Error starting analysis: ${error.message}`);
            }
        },
        
        // Simple polling instead of WebSocket
        startPolling() {
            // Poll every 3 seconds
            if (this.pollInterval) {
                clearInterval(this.pollInterval);
            }
            
            this.pollInterval = setInterval(async () => {
                try {
                    const response = await fetch('/api/analysis/status');
                    const status = await response.json();
                    
                    // Update state
                    this.analysisState = status;
                    
                    // If analysis just completed, refresh stats
                    if (!status.is_running && status.progress_pct === 100 && !this.analysisCompleted) {
                        this.analysisCompleted = true;
                        await this.loadConfig();
                        await this.loadAllSongs();
                        
                        // Stop polling since analysis is done
                        this.stopPolling();
                        
                        // Reset flag after 5 seconds
                        setTimeout(() => {
                            this.analysisCompleted = false;
                        }, 5000);
                    }
                    
                    // Stop polling if analysis is not running (idle state)
                    if (!status.is_running && status.progress_pct === 0) {
                        this.stopPolling();
                    }
                } catch (error) {
                    console.error('Error polling status:', error);
                }
            }, 3000); // Poll every 3 seconds
        },
        stopPolling() {
            if (this.pollInterval) {
                clearInterval(this.pollInterval);
                this.pollInterval = null;
            }
        },
        
        async checkAnalysisStatus() {
            try {
                const response = await fetch('/api/analysis/status');
                const status = await response.json();
                this.analysisState = status;
                
                // Only start polling if analysis is actively running
                if (status.is_running) {
                    this.startPolling();
                }
            } catch (error) {
                console.error('Error checking status:', error);
            }
        },
        
        // Text Search Methods
        // Text Search Methods
        async searchByText() {
            if (!this.textQuery.trim()) {
                alert('Please enter a search query');
                return;
            }
            
            // Check if songs are analyzed
            if (this.stats.total_songs === 0) {
                alert('⚠️ No songs analyzed yet! Please go to the Analysis tab and analyze your songs first.');
                return;
            }
            
            this.searching = true;
            this.textResults = [];
            
            try {
                const response = await fetch('/api/search/text', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        query: this.textQuery,
                        limit: 20 
                    })
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Search failed');
                }
                
                const data = await response.json();
                this.textResults = data.results;
                
                if (this.textResults.length === 0) {
                    alert('No results found for your query. Try a different search term.');
                }
                
                // Update stats if we ran new analysis
                if (!data.cached) {
                    await this.loadConfig();
                }
            } catch (error) {
                alert(`❌ Error searching: ${error.message}`);
            } finally {
                this.searching = false;
            }
        },
        
        // Similarity Search Methods
        async loadAllSongs() {
            try {
                const response = await fetch('/api/songs');
                const data = await response.json();
                this.allSongs = data.songs;
                this.filteredSongs = data.songs;
            } catch (error) {
                console.error('Error loading songs:', error);
            }
        },
        
        filterSongs() {
            if (!this.songSearchQuery.trim()) {
                this.filteredSongs = this.allSongs;
                return;
            }
            
            const query = this.songSearchQuery.toLowerCase();
            this.filteredSongs = this.allSongs.filter(song => {
                const artist = (song.artist || '').toLowerCase();
                const title = (song.title || song.filename).toLowerCase();
                return artist.includes(query) || title.includes(query);
            });
        },
        
        async searchSimilar() {
            if (!this.selectedSongId) {
                return;
            }
            
            // Check if songs are analyzed
            if (this.stats.total_songs === 0) {
                alert('⚠️ No songs analyzed yet! Please go to the Analysis tab and analyze your songs first.');
                return;
            }
            
            if (this.stats.total_songs < 2) {
                alert('⚠️ Need at least 2 songs to find similar tracks!');
                return;
            }
            
            this.similarResults = [];
            
            try {
                const response = await fetch('/api/search/similar', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        song_id: parseInt(this.selectedSongId),
                        limit: 20 
                    })
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Search failed');
                }
                
                const data = await response.json();
                this.similarResults = data.results;
                
                if (this.similarResults.length === 0) {
                    alert('No similar songs found.');
                }
            } catch (error) {
                alert(`❌ Error finding similar songs: ${error.message}`);
            }
        },
        
        // Audio Player Methods
        playSong(songId, playerId = 'audioPlayer') {
            const player = document.getElementById(playerId);
            
            if (this.currentlyPlaying === songId) {
                // Pause
                player.pause();
                player.src = '';
                this.currentlyPlaying = null;
            } else {
                // Play new song
                this.currentlyPlaying = songId;
                this.$nextTick(() => {
                    player.src = `/api/audio/${songId}`;
                    player.play();
                });
            }
        },
        
        getCurrentSongInfo() {
            if (!this.currentlyPlaying) return '';
            
            // Find song in all results
            let song = this.textResults.find(s => s.id === this.currentlyPlaying);
            if (!song) song = this.similarResults.find(s => s.id === this.currentlyPlaying);
            if (!song) song = this.allSongs.find(s => s.id === this.currentlyPlaying);
            
            if (song) {
                return `🎵 Now Playing: ${song.artist || 'Unknown'} - ${song.title || song.filename}`;
            }
            return '🎵 Now Playing';
        },
        
        // Utility Methods
        formatDuration(seconds) {
            if (!seconds) return '0h';
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            
            if (hours > 0) {
                return `${hours}h ${minutes}m`;
            } else {
                return `${minutes}m`;
            }
        }
    }
}
