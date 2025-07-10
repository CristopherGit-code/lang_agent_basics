from dotenv import load_dotenv
load_dotenv()
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langgraph.graph import StateGraph, START, MessagesState
from langchain_core.runnables import RunnableSequence,chain
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from agents.src.utils.config import Settings
from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool
from typing import Annotated, List
import logging, os
from langgraph.types import Command,Send

checkpointer = InMemorySaver()

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

# TODO: use different ollama models

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

verify_model = ChatOCIGenAI(
    model_id="cohere.command-a-03-2025",
    service_endpoint=settings.oci_client.endpoint,
    compartment_id=settings.oci_client.compartiment,
    model_kwargs={"temperature":0.85, "max_tokens":500},
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

verify_agent = create_react_agent(
    model=verify_model,
    tools=[],
    prompt=(
        "You are an agent specialized on verify the response given to the user\n"
        "Respond as if you are the agent in charge of do the workflow execution, not in third person, consider ONLY the information provided in the response\n"
        "if there is an error in response, help the user to navigate to a better usage and if not, mantain as much as possible from the original response\n"
        "Answer considering the user has no clue of the internal thinking process or any other details extra from your response, so consider provide a good context and a good polite and friendly response\n"
        "Keep your explanation BELOW 50 words\n"
    ),
    name="verify_agent"
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
            "Description of what the next agent should do, including all of the relevant context. Include necessary arguments for the next agent to work at the best performance possible",
        ],
        # these parameters are ignored by the LLM, add manually
        state: Annotated[MessagesState, InjectedState],
    ) -> Command:
        extra_args = state["messages"][-1].tool_calls[0]['args']['state']
        supervisor_instruction = task_description + f" extra details: {extra_args}"
        # logger.debug(supervisor_instruction)
        task_description_message = {"role": "user", "content": supervisor_instruction}
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
        "-After compleating the workflow, respond the user query from the beginning with the information from the agents\n"
        "At the last user interaction after all the workflow is done, offer the user the message indicating the result of the work and some bakcground information of what you did"
    ),
    name="supervisor"
)

# ------------- final verify -----------------------

main_graph_builder = StateGraph(MessagesState)
main_graph_builder.add_node(supervisor_agent)
main_graph_builder.add_node(file_agent)
main_graph_builder.add_node(song_agent)
# main_graph_builder.add_node("verify",verify)
main_graph_builder.add_edge(START, "supervisor")
# always return back to the supervisor
main_graph_builder.add_edge("file_agent", "supervisor")
main_graph_builder.add_edge("song_agent", "supervisor")
supervisor = main_graph_builder.compile(checkpointer=checkpointer)

# ------------- chat interface ---------------------
@chain
def stream_updates(user_input: str) -> List[str]:
    """Creates a custom workflow to fulfill the user's request using a variety of agents"""
    final_response = []
    try:
        for chunk in supervisor.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            {"configurable": {"thread_id": "1"}},
            stream_mode="values",
            subgraphs=True
        ):
            logger.debug("=============== current chunk ===============")
            logger.debug(chunk[-1]['messages'][-1])
            try:
                logger.debug(chunk[-1]['messages'][-1].tool_calls)
            except Exception:
                logger.debug("No tool calls for this step")
            try:
                final_response.append(chunk[-1]['messages'][-1].content)
            except Exception as p:
                final_response.append(f"Error in response: {p}")
    except Exception as e:
        logger.debug(e)
        final_response.append(f"Error in response: {e}")
    response = [final_response[-1],user_input]
    return response

@chain
def verify_response(response:List[str])->str:
    logger.info("=============== started verification ===============")
    final_response = []
    logger.debug(response[0])
    logger.debug(response[1])
    info = f"Here is the query:{response[1]} and the response: {response[0]}"
    for chunk in verify_agent.stream(
        {"messages": [{"role": "user", "content": info}]},
        {"configurable": {"thread_id": "1"}},
        stream_mode="values",
        subgraphs=True
    ):
        logger.debug(chunk[-1])
        try:
            final_response.append(chunk[-1]['messages'][-1].content)
        except Exception as p:
            final_response.append(f"Error in response: {p}")
    return final_response[-1]

my_chain = RunnableSequence(stream_updates,verify_response)

while True:
    try:
        user_input = input("USER: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        response = my_chain.invoke(user_input)
        print("AGENT:\n",response)
    except Exception as e:
        logger.info(f'General error: {e}')