from typing import Annotated, Literal, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from PIL import Image
import matplotlib.pyplot as plt
import io

from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Wikipedia検索ツールを定義
wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=700)
# WikipediaQueryRunを直接使用
wikipedia_tool_base = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

# DuckDuckGoウェブ検索ツールを定義
duckduckgo_search = DuckDuckGoSearchRun()

# --- ツールの定義 (説明文を英語に統一) ---
@tool
def search_wikipedia(query: str) -> str:
    """Use this tool for searching content that has a dedicated Wikipedia article, 
    such as general knowledge, historical figures, or academic topics.
    Prioritize this over general web search."""
    return wikipedia_tool_base.run(query)

@tool
def search_web(query: str) -> str:
    """Search the web using DuckDuckGo."""
    return duckduckgo_search.run(query)

tools = [search_web]

tool_node = ToolNode(tools)

model = ChatOllama(model="qwen2.5:7b-instruct", base_url="http://localhost:11434").bind_tools(tools)

# model = ChatOpenAI(api_key=config.OPENAI_API_KEY,model_name="gpt-4o-mini").bind_tools(tools)


# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", 'agent')

checkpointer  = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=checkpointer )

thread = {"configurable":{"thread_id":"42"}}
inputs = [HumanMessage(content="what is the weather in sf")]
for event in app.stream({"messages":inputs},thread,stream_mode="values"):
    event["messages"][-1].pretty_print()
