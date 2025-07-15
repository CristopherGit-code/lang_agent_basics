from langchain_core.tools import tool
import os

class FileTool:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FileTool,cls).__new__(cls)
        return cls._instance

    @tool
    def _write_file(path:str, content:str)->str:
        """Writes content into a file path, could be provided or default, current directory"""
        llm_data = str(content)
        with open(path,'w') as file:
            data = "\n"+llm_data
            file.write(data)
        return "File written successfully"

    @tool
    def _delete_file(path:str)->str:
        """Deletes a file given a path from the user"""
        os.remove(path)
        return f"File deleted at {path}"

    @tool
    def _open_file(path:str)->str:
        """Reads the content of a file given a path by the user"""
        with open(path,'r') as file:
            content = file.read()
            return f'File content:\n{content}'
    
    def get_file_tools(self):
        file_tools = [
            self._write_file,
            self._delete_file,
            self._open_file
        ]
        return file_tools