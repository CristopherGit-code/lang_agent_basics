import uuid
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from src.modules.config.config import Settings

class FuseConfig:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FuseConfig,cls).__new__(cls)
            cls._instance._init()
        return cls._instance
    
    def _init(self):
        if self._initialized:
            return
        self.settings = Settings(r"C:\Users\Cristopher Hdz\Desktop\Test\hw_agent\src\modules\config\agent.yaml")        
        Langfuse(
            public_key=self.settings.langfuse.public_key,
            secret_key=self.settings.langfuse.secret_key,
            host="https://cloud.langfuse.com",
        )
        self.langfuse_handler = CallbackHandler()
        self._initialized = True

    def get_handler(self):
        return self.langfuse_handler
    
    def generate_id(self)->str:
        return str(uuid.uuid4())