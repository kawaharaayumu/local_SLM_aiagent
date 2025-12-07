import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
import uvicorn

# OllamaをLLMとして設定
llm = OllamaLLM(model="llama3:8b-instruct-q4_K_M", base_url="http://localhost:11434")

# Wikipedia検索ツールを定義
wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=700)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

# DuckDuckGoウェブ検索ツールを定義
duckduckgo_search = DuckDuckGoSearchRun()                     

# --- ツールの説明文を工夫して、優先順位をLLMに伝える ---
@tool
def search_wikipedia(query: str) -> str:
    """一般的な知識、歴史上の人物、学術的なトピックなど、Wikipediaに記事がある内容を検索するのに使います。
    ウェブ検索よりも優先して利用してください。"""
    return wikipedia_tool.run({"query": query})

@tool
def search_web(query: str) -> str:
    """Wikipediaに情報がない、最新のニュースや出来事、特定の商品情報など、より広範なウェブ検索が必要な場合に利用してください。"""
    return duckduckgo_search.run(query)

# エージェントが利用できるツールをリストにまとめる
# ここにツールの優先順位はありません。LLMが説明文を読んで判断します。
tools = [search_wikipedia, search_web]

# エージェントの初期化
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True # この行を追加
)

# FastAPIのインスタンスを作成
app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat_with_llama3(request: PromptRequest):
    try:
        result = agent.run(request.prompt)
        return {"response": result}
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)