import logging
from datetime import datetime
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
wikipedia_tool_base = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)
duckduckgo_search = DuckDuckGoSearchRun()

@tool
def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for a specific topic. 
    Input should be a focused search query (e.g., a specific name, place, or concept).
    """
    return wikipedia_tool_base.run(query)

@tool
def search_web(query: str) -> str:
    """Search the web using DuckDuckGo."""
    return duckduckgo_search.run(query)

from datetime import datetime

@tool
def get_datetime_now() -> str:
    """
    Returns the current date, time, and day of the week in 'YYYY-MM-DD HH:MM:SS (Day)' format.
    Useful for accurate time tracking and scheduling.
    """
    # 曜日のリスト（0=月曜日, 6=日曜日）
    weeks = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    day_name = weeks[now.weekday()]
    
    return f"{time_str} ({day_name})"

# tool
# tools = [search_web, search_wikipedia]
tools = [search_wikipedia,get_datetime_now]