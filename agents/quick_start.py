from dotenv import load_dotenv
load_dotenv()
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from src.utils.config import Settings
from langgraph.prebuilt import create_react_agent
import logging
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(name=f"Agent.{__name__}")

settings = Settings(r"C:\Users\Cristopher Hdz\Desktop\Test\hw_agent\src\config\hw_agent.yaml")

class PrettyResponse(BaseModel):
    conditions:str

# enables short-term memory. 
# Stores the state of the agent at each step in the tool calling loop
checkpointer = InMemorySaver()

# context from the state to give the model
def dynamic_prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:  
    user_name = config["configurable"].get("user_name")
    system_msg = f"You are a helpful assistant. Address the user as {user_name}."
    return [{"role": "system", "content": system_msg}] + state["messages"]

# Requires comment description of the function
# tool that the agent could use
def get_weather(city:str)->str:
    """Gets the weather for a given city"""
    return f"It is sunny in {city}"

llm = ChatOCIGenAI(
    model_id="cohere.command-a-03-2025",
    service_endpoint=settings.oci_client.endpoint,
    compartment_id=settings.oci_client.compartiment,
    model_kwargs={"temperature":0.7, "max_tokens":500},
    auth_profile=settings.oci_client.configProfile,
    auth_file_location=settings.oci_client.config_path
)

# Static prompt as system instructions message
static_prompt = "You are a helpful assistant"

# prompt parameter is put before the actual user query
# response format adds an step at the end of the loop. Generates a new llm call to format the response 
# response_format=("Respond in a friendly tone",PrettyResponse) not supported by oci
agent = create_react_agent(
    model=llm,
    tools=[get_weather],
    prompt=dynamic_prompt,
    checkpointer=checkpointer,
)
logger.info('Agent created')

# Runnable object for config state and memory
config = {
    "configurable":{
        "user_name":"Cristopher Hernandez",
        "thread_id":"invoke"
    }
}

static_prompt_response = agent.invoke(
    {"messages":[{"role":"user","content":"What is the weather in sf?"}]},
    config=config
)
print(f"Model response:\n{static_prompt_response}\n\n")

# state["messages"] access to the first argument in function
dynamic_prompt_response = agent.invoke(
    {"messages":[{"role":"user","content":"What is the weather in ny?"}]},
    config=config
)
print(f"Model response:\n{dynamic_prompt_response}")

memory_response = agent.invoke(
    {"messages":[{"role":"user","content":"Which was my first request?"}]},
    config=config
)
print(f"Model response:\n{memory_response["structured_response"]}")