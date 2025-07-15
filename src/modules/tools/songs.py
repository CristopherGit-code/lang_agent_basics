from langchain_core.tools import tool

class SongTool:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SongTool,cls).__new__(cls)
        return cls._instance

    @tool
    def _get_song_data(name:str)->str:
        """Returns the information from a given song name"""
        if len(name) > 10:
            return f"{name} is a classic song, pretty good"
        else:
            return f"{name} is from the new states, great choice!"
        
    @tool
    def _play_song(name:str,user_name:str)->str:
        """Plays a song into the personal sound system of the user_name"""
        return f"{name} now playing at speakers of {user_name}"

    @tool
    def _queue_song(name:str)->str:
        """Adds the current name to the playlist queue"""
        return f"Song: {name} added to the playlist"
    
    def get_song_tools(self):
        song_tools = [
            self._get_song_data,
            self._play_song,
            self._queue_song
        ]
        return song_tools