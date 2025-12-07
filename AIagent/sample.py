import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

# --- アプリケーション起動時に一度だけ実行される処理 ---

# OllamaをLLMとして設定
llm = OllamaLLM(model="llama3:8b-instruct-q4_K_M", base_url="http://localhost:11434")
# DuckDuckGoを検索ツールとして定義
@tool
def search_duckduckgo(query: str) -> str:
    """ウェブ検索が必要な質問に答えるために使います。天気、ニュース、現在の情報など、最新の情報を探すのに役立ちます。"""
    search = DuckDuckGoSearchRun()
    return search.run(query)

tools = [search_duckduckgo]

# エージェントの初期化
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# FastAPIのインスタンスを作成
app = FastAPI()

# --- APIリクエストごとに実行される処理 ---

# APIリクエストのボディを定義するためのPydanticモデル
class PromptRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat_with_llama3(request: PromptRequest):
    try:
        # 起動時に作成されたagentインスタンスを再利用
        result = agent.run(request.prompt)

        return {"response": result}

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}