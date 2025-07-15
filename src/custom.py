from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool
from typing import Annotated
from langgraph.types import Command,Send
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from .modules.llm_client import LLM_Client
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import Runnable
from .modules.graphs import State
from .modules.graphs import GraphAgent

class CustomAgent:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CustomAgent,cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        if self._initialized:
            return
        self.llm_client = LLM_Client()        
        self.checkpointer = InMemorySaver()
        self.graph_agents = GraphAgent()
        self.config = {"configurable":{"thread_id":"invoke"}}
        self.supervisor_tools = self.build_supervisor_tools()
        self.supervisor = self.build_supervisor_agent()
        self._initialized = True

    def supervisor_chatbot(self,state:State):
        return {"messages": [self.supervisor_agent_with_tools.invoke(state["messages"])]}
    
    def _create_task_description_handoff_tool(self,
            *, agent_name: str, description: str | None = None
        ):
        name = f"transfer_to_{agent_name}"
        description = description or f"Ask {agent_name} for help."

        @tool(name, description=description)
        def _handoff_tool(
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

        return _handoff_tool

    def build_supervisor_tools(self):
        _assign_to_file_agent = self._create_task_description_handoff_tool(
            agent_name="file_agent",
            description="Assign task to the file manager agent",
        )
        _assign_to_song_agent = self._create_task_description_handoff_tool(
            agent_name="song_agent",
            description="Assign task to the song manager agent",
        )
        return [_assign_to_file_agent,_assign_to_song_agent]
    
    def build_supervisor_agent(self):
        llm = self.llm_client.build_llm_client()
        self.supervisor_agent_with_tools = llm.bind_tools(self.supervisor_tools)
        graph_builder = StateGraph(State)
        graph_builder.add_node("supervisor_bot", self.supervisor_chatbot)
        supervisor_tool_node = ToolNode(tools=self.supervisor_tools)
        graph_builder.add_node("supervisor_tools", supervisor_tool_node)
        graph_builder.add_conditional_edges(
            "supervisor_bot",
            self.graph_agents.route_tools,
            {"tools": "supervisor_tools", END: END},
        )
        graph_builder.add_edge("supervisor_tools","supervisor_bot")
        graph_builder.add_edge(START, "supervisor_bot")
        supervisor_agent = graph_builder.compile(name="supervisor_agent")
        return supervisor_agent