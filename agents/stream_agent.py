from dotenv import load_dotenv
load_dotenv()
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from src.utils.config import Settings
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import logging, os, json
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(name=f"Agent.{__name__}.-------->")
settings = Settings(r"C:\Users\Cristopher Hdz\Desktop\Test\hw_agent\src\config\hw_agent.yaml")
config = {"configurable": {"thread_id": "1"}}

# --------------- tool section -----------------------
@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

@tool
def write_file(path:str, content:str)->str:
    """Writes content into a file path, could be provided or default, current directory"""
    llm_data = str(content)
    with open(path,'w') as file:
        data = "\n"+llm_data
        file.write(data)
    return "File written successfully"

@tool
def delete_file(path:str)->str:
    """Deletes a file given a path from the user"""
    os.remove(path)
    return f"File deleted at {path}"

@tool
def open_file(path:str)->str:
    """Reads the content of a file given a path by the user"""
    with open(path,'r') as file:
        content = file.read()
        return f'File content:\n{content}'
    
tools = [write_file,delete_file,open_file]

# ----------------------------------------------------

# First state in graph build
class State(TypedDict):
    # defines how the state key is updated (ej. append messages, rather than overwritting)
    # keys without a reducer overwrite past values per default
    messages: Annotated[list, add_messages]

# store in run, change for sqlite or TODO: custom oracle DB checkpointer
checkpointer = MemorySaver()

llm = ChatOCIGenAI(
    model_id="cohere.command-a-03-2025",
    service_endpoint=settings.oci_client.endpoint,
    compartment_id=settings.oci_client.compartiment,
    model_kwargs={"temperature":0.7, "max_tokens":500},
    auth_profile=settings.oci_client.configProfile,
    auth_file_location=settings.oci_client.config_path
)

llm_with_tools = llm.bind_tools(tools)

# function node for the graph. 
# Uses and returns the dictionary with the messages key
def chatbot(state:State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Tracks the state of the complete loop
graph_builder = StateGraph(State)
# id of node, function/object called when node is used
graph_builder.add_node("chatbot",chatbot)

# ------------------ tool call function --------------

# prebuild for the class BasicToolNode
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# ----------------------------------------------------

# Prebuilt for tool function verification
graph_builder.add_conditional_edges(
    "chatbot",tools_condition
)

# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
# entry point, tells the graph where to start each time
graph_builder.add_edge(START, "chatbot")
# compiled state graph to invoke with state
graph = graph_builder.compile(checkpointer=checkpointer)

# ------------- chat interface ---------------------
def stream_updates(user_input: str):
    # stream equivalent of streaming graph.invoke
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        stream_mode="updates",
        subgraphs=True
    ):
        # print("Assistant:",chunk['messages'][-1].content)
        print("\n\nChunk response:",chunk,"\n\n")

# Not completely useful
def stream_tokens(user_input:str):
    for token, metadata in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        stream_mode="messages",
        subgraphs=True
    ):
        print("\n\nChunk response:",token,"\n",metadata)

def stream_multiple(user_input: str):
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        stream_mode=["updates","values","checkpoints"],
        subgraphs=True
    ):
        print("\n\nChunk response:",chunk,"\n\n")

def stream_debug(user_input: str):
    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        stream_mode="debug",
        subgraphs=True
    ):
        print("\n\nChunk response:",chunk,"\n\n")
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        # stream_updates(user_input)
        # stream_tokens(user_input)
        # stream_multiple(user_input)
        stream_debug(user_input)
    except:
        # in case input or stream fails
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_updates(user_input)
        break