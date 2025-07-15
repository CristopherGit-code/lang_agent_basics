from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from .llm_client import LLM_Client
from .tools.files import FileTool
from langgraph.checkpoint.memory import InMemorySaver
from .tools.songs import SongTool
from langchain_core.runnables import Runnable

class State(TypedDict):
    messages: Annotated[list, add_messages]

class GraphAgent:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GraphAgent,cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        if self._initialized:
            return
        self.llm_client = LLM_Client()
        self.song_tools = SongTool().get_song_tools()
        self.file_tools = FileTool().get_file_tools()
        self.checkpointer = InMemorySaver()
        self.config = {"configurable":{"thread_id":"invoke"}}
        self.file_agent = self.build_file_agent()
        self.song_agent = self.build_song_agent()
        self._initialized = True

    def route_tools(self,state: State,):
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return END

    def file_chatbot(self,state:State):
        return {"messages": [self.file_agent_with_tools.invoke(state["messages"])]}
    
    def song_chatbot(self,state:State):
        return {"messages": [self.song_agent_with_tools.invoke(state["messages"])]}
    
    def build_file_agent(self):
        llm = self.llm_client.build_llm_client()
        self.file_agent_with_tools = llm.bind_tools(self.file_tools)
        graph_builder = StateGraph(State)
        graph_builder.add_node("file_bot", self.file_chatbot)
        file_tool_node = ToolNode(tools=self.file_tools)
        graph_builder.add_node("file_tools", file_tool_node)
        graph_builder.add_conditional_edges(
            "file_bot",
            self.route_tools,
            {"tools": "file_tools", END: END},
        )
        graph_builder.add_edge("file_tools","file_bot")
        graph_builder.add_edge(START, "file_bot")
        file_agent = graph_builder.compile(name="file_agent")
        return file_agent
    
    def build_song_agent(self):
        llm = self.llm_client.build_llm_client()
        self.song_agent_with_tools = llm.bind_tools(self.song_tools)
        graph_builder = StateGraph(State)
        graph_builder.add_node("song_bot", self.song_chatbot)
        song_tool_node = ToolNode(tools=self.song_tools)
        graph_builder.add_node("song_tools", song_tool_node)
        graph_builder.add_conditional_edges(
            "song_bot",
            self.route_tools,
            {"tools": "song_tools", END: END},
        )
        graph_builder.add_edge("song_tools","song_bot")
        graph_builder.add_edge(START, "song_bot")
        song_agent = graph_builder.compile(name="song_agent")
        return song_agent