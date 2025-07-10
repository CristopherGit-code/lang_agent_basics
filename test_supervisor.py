from dotenv import load_dotenv
load_dotenv()
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from agents.src.utils.config import Settings
from typing import Annotated
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import InjectedState
import logging, os
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(name=f"Agent.{__name__}.-------->")
settings = Settings(r"C:\Users\Cristopher Hdz\Desktop\Test\hw_agent\agents\src\config\hw_agent.yaml")

# --------------- tool section -----------------------
@tool
def get_song_data(name:str)->str:
    """Returns the information from a given song name"""
    if len(name) > 10:
        return f"{name} is a classic song, pretty good"
    else:
        return f"{name} is from the new states, great choice!"
    
@tool
def play_song(name:str,user_name:str)->str:
    """Plays a song into the personal sound system of the user_name"""
    return f"{name} now playing at speakers of {user_name}"

@tool
def queue_song(name:str)->str:
    """Adds the current name to the playlist queue"""
    return f"Song: {name} added to the playlist"

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
    
file_tools = [write_file,delete_file,open_file]
song_tools = [get_song_data,play_song,queue_song]

file_model = ChatOCIGenAI(
    model_id="cohere.command-a-03-2025",
    service_endpoint=settings.oci_client.endpoint,
    compartment_id=settings.oci_client.compartiment,
    model_kwargs={"temperature":0.7, "max_tokens":500},
    auth_profile=settings.oci_client.configProfile,
    auth_file_location=settings.oci_client.config_path
)

song_model = ChatOCIGenAI(
    model_id="cohere.command-a-03-2025",
    service_endpoint=settings.oci_client.endpoint,
    compartment_id=settings.oci_client.compartiment,
    model_kwargs={"temperature":0.7, "max_tokens":500},
    auth_profile=settings.oci_client.configProfile,
    auth_file_location=settings.oci_client.config_path
)

supervisor_model = ChatOCIGenAI(
    model_id="cohere.command-a-03-2025",
    service_endpoint=settings.oci_client.endpoint,
    compartment_id=settings.oci_client.compartiment,
    model_kwargs={"temperature":0.7, "max_tokens":500},
    auth_profile=settings.oci_client.configProfile,
    auth_file_location=settings.oci_client.config_path
)

file_agent = create_react_agent(
    model=file_model,
    tools=file_tools,
    prompt=(
        "You are a file system manager agent"
        "Instructions:\n"
        "-Assist only with file management tasks\n"
        "-When finish, respond to supervisor directly\n"
        "-Respond ONLY with the results of your work, do NOT include ANY extra text"
    ),
    name="file_agent"
)

song_agent = create_react_agent(
    model=song_model,
    tools=song_tools,
    prompt=(
        "You are a song manager agent"
        "Instructions:\n"
        "-Assist only with song management tasks\n"
        "-When finish, respond to supervisor directly\n"
        "-Respond ONLY with the results of your work, do NOT include ANY extra text"
    ),
    name="song_agent"
)

# --------------- supervisor -----------------------

## ===============> Handsoff management

def create_task_description_handoff_tool(
    *, agent_name: str, description: str | None = None
):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        # this is populated by the supervisor LLM
        task_description: Annotated[
            str,
            "Description of what the next agent should do, including all of the relevant context.",
        ],
        # these parameters are ignored by the LLM
        state: Annotated[MessagesState, InjectedState],
    ) -> Command:
        task_description_message = {"role": "user", "content": task_description}
        agent_input = {**state, "messages": [task_description_message]}
        return Command(
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )

    return handoff_tool

assign_to_file_agent = create_task_description_handoff_tool(
    agent_name="file_agent",
    description="Assign task to the file manager agent",
)
assign_to_song_agent = create_task_description_handoff_tool(
    agent_name="song_agent",
    description="Assign task to the song manager agent",
)

## ===============> Supervisor (simplified to avoid graph core creation)

supervisor_agent = create_react_agent(
    model=supervisor_model,
    tools=[assign_to_file_agent,assign_to_song_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a file agent. Assign file_management-related tasks to this agent\n"
        "- a song agent. Assign song-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "-After compleating the workflow, respond the user query from the beginning"
    ),
    name="supervisor"
)

checkpointer = MemorySaver()

supervisor = (
    StateGraph(MessagesState)
    .add_node(supervisor_agent)
    .add_node(file_agent)
    .add_node(song_agent)
    .add_edge(START, "supervisor")
    # always return back to the supervisor
    .add_edge("file_agent", "supervisor")
    .add_edge("song_agent", "supervisor")
    .compile()
)

config = {"configurable":{"thread_id":"main_supervisor"}}

# ------------- chat interface ---------------------
def stream_updates(user_input: str):
    try:
        for chunk in supervisor.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="values"
        ):
            logger.debug(chunk[-1])
            print("=============== AGENT RESPONSE:",chunk['messages'][-1].content)
    except Exception as e:
        logger.debug(e)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_updates(user_input)
    except Exception as e:
        logger.info(f'General error: {e}')