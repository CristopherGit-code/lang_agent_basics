from .config.config import Settings
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI

class LLM_Client:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLM_Client,cls).__new__(cls)
            cls._instance._init()
        return cls._instance
    
    def _init(self):
        if self._initialized:
            return
        self.settings = Settings(r"C:\Users\Cristopher Hdz\Desktop\Test\hw_agent\src\modules\config\agent.yaml")
        self._initialized = True

    def build_llm_client(self):
        llm = ChatOCIGenAI(
            model_id="cohere.command-a-03-2025",
            service_endpoint=self.settings.oci_client.endpoint,
            compartment_id=self.settings.oci_client.compartiment,
            model_kwargs={"temperature":0.7, "max_tokens":self.settings.oci_client.max_tokens},
            auth_profile=self.settings.oci_client.configProfile,
            auth_file_location=self.settings.oci_client.config_path
        )
        return llm

def main():
    llm = LLM_Client()
    llm_client = llm.build_llm_client()

if __name__ == "__main__":
    main()