from src.prebuild import PreBuildAgent
from src.custom import CustomAgent
from src.modules.graphs import GraphAgent
from src.modules.fuse_config import FuseConfig
from langgraph.graph import StateGraph, START, MessagesState
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(name=f"Agent.{__name__}")

prebuild_agents = PreBuildAgent()
custom_agents = CustomAgent()
graph_agents = GraphAgent()
fuse_config = FuseConfig()
id = fuse_config.generate_id()

## Prebuild libraries
main_graph_builder = StateGraph(MessagesState)
main_graph_builder.add_node(prebuild_agents.supervisor_agent)
for agent in prebuild_agents.workers:
    main_graph_builder.add_node(agent[1])
    logger.debug(f"Add agent: {agent[0]}")
main_graph_builder.add_edge(START, "supervisor")
for agent in prebuild_agents.workers:
    main_graph_builder.add_edge(agent[0],"supervisor")
supervisor = main_graph_builder.compile(checkpointer=prebuild_agents.checkpointer)

## Manual agent build
""" main_graph_builder = StateGraph(MessagesState)
main_graph_builder.add_node(custom_agents.supervisor)
main_graph_builder.add_node(graph_agents.song_agent)
main_graph_builder.add_node(graph_agents.file_agent)
main_graph_builder.add_edge(START, "supervisor_agent")
main_graph_builder.add_edge("file_agent", "supervisor_agent")
main_graph_builder.add_edge("song_agent", "supervisor_agent")
supervisor = main_graph_builder.compile(checkpointer=custom_agents.checkpointer)
logger.info("Custom supervisor built") """

def stream_updates(user_input: str) -> str:
    """Creates a custom workflow to fulfill the user's request using a variety of agents"""
    final_response = []
    try:
        for chunk in supervisor.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            {"configurable": {"thread_id": "1"},"callbacks":[fuse_config.get_handler()], "metadata":{"langfuse_session_id":id}},
            stream_mode="values",
            subgraphs=True
        ):
            logger.debug("=============== current chunk ===============")
            logger.debug(chunk[-1]['messages'][-1])
            try:
                final_response.append(chunk[-1]['messages'][-1].content)
            except Exception as p:
                final_response.append(f"Error in response: {p}")
    except Exception as e:
        logger.debug(e)
        final_response.append(f"Error in response: {e}")
    return final_response[-1]

while True:
    try:
        user_input = input("USER: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        response = stream_updates(user_input)
        print("AGENT:\n",response)
    except Exception as e:
        logger.info(f'General error: {e}')