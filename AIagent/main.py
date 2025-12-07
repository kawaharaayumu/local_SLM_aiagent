import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import time
import hashlib

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.agents import  AgentFinish
# 修正: toolsのインポートをLangChainの推奨パスに変更
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from typing import Annotated, Literal, TypedDict

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

# LangGraph関連のインポート
from langgraph.graph import StateGraph, END

import logging
import sys

# ロガーを取得
logger = logging.getLogger(__name__) 
# ログレベルを設定（ここではデバッグレベルから全て出力）
logger.setLevel(logging.DEBUG)

# 既にハンドラーが設定されている場合は重複を避ける
if not logger.handlers:
    # 標準出力(sys.stdout)にログを出力するハンドラーを作成
    handler = logging.StreamHandler(sys.stdout)
    # ハンドラーのレベルを設定
    handler.setLevel(logging.DEBUG) 
    # フォーマッターを設定（Uvicornのデフォルトに近い形式）
    formatter = logging.Formatter('%(levelname)s:     %(name)s - %(message)s')
    handler.setFormatter(formatter)
    
    # ロガーにハンドラーを追加
    logger.addHandler(handler)
    


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


# FastAPIのインスタンスを作成
app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/agent")
async def agent_by_langgraph(request: PromptRequest):
    """
    LangGraphベースのエージェントにプロンプトを送信し、応答を取得します。
    ストリーム処理の途中経過を logger.info で出力します。
    """
    # 既存のグローバル変数から取得
    global workflow, checkpointer 
    
    logger.info("--- API Request Received ---")
    
    try:
        # LangGraphをコンパイル（元のコードに合わせて関数内で実行）
        lang_app = workflow.compile(checkpointer=checkpointer)
        
        # リクエストから一意のスレッドIDを生成 (簡易的な実装)
        thread_id = hashlib.sha256(f"{request.prompt}_{time.time()}".encode()).hexdigest()
        thread = {"configurable":{"thread_id": thread_id}}
        
        # ユーザーからのプロンプトをHumanMessageとして設定
        inputs = [HumanMessage(content=request.prompt)]
        
        final_response_message = None
        
        logger.info(f"Starting LangGraph stream (Thread ID: {thread_id}, Prompt: {request.prompt[:50]}...)")
        
        # ストリームを実行し、中間ステップのログを出力
        for i, event in enumerate(lang_app.stream({"messages":inputs}, thread, stream_mode="values")):
            last_message = event.get("messages", [None])[-1]

            if not last_message:
                continue

            # -----------------------------------------------------
            # ⭐ 思考の途中経過を logger.info で出力するロジック ⭐
            # -----------------------------------------------------
            log_output = ""
            
            if isinstance(last_message, AIMessage):
                # エージェントからの応答/思考/ツール呼び出し
                if last_message.tool_calls:
                    # ツール呼び出し（思考とアクション）
                    tool_calls_str = ", ".join([
                        f"{tc['name']}(args={repr(tc['args'])})" # reprで引数を明確に
                        for tc in last_message.tool_calls
                    ])
                    log_output = f"[AGENT] 🛠️ Tool Call: {tool_calls_str}"
                elif last_message.content:
                    # 最終的な回答、または中間的な思考（コンテンツのみ）
                    log_output = f"[AGENT] 💬 Response/Thought: {last_message.content.strip()[:100]}..." # 先頭100文字
                    # ツール呼び出しがないAIMessageを最終応答として記録
                    final_response_message = last_message 
                
            elif isinstance(last_message, ToolMessage):
                # ツールの実行結果
                content_preview = last_message.content.strip()[:100].replace('\n', ' ')
                tool_name = f"'{last_message.name}'" if last_message.name else "N/A"
                log_output = f"[TOOL] ✅ Result from {tool_name}: {content_preview}..."
            
            elif isinstance(last_message, HumanMessage):
                # 最初のインプット (1回目のみログ)
                if i == 0:
                     log_output = f"[INPUT] 👤 User Prompt: {last_message.content.strip()[:100]}..."
            
            if log_output:
                 # INFOレベルで出力
                 logger.info(f"[{i:02}] {log_output}")
            # -----------------------------------------------------
            
        logger.info("LangGraph stream finished successfully. Retrieving final response.")

        # 最終応答の抽出とフォールバック
        response_text = "処理は完了しましたが、最終的なテキスト応答が見つかりませんでした。"
        if final_response_message and final_response_message.content:
            response_text = final_response_message.content
        elif event.get("messages") and isinstance(event["messages"][-1], AIMessage):
            # 念のため、最後のAIMessageをフォールバックとして使用
            response_text = event["messages"][-1].content
                      
        logger.info(f"Final Response Sent (Length: {len(response_text)} chars)")
                 
        return {"response": response_text}
        
    except Exception as e:
        # エラー時のログ
        logger.error(f"An error occurred during agent execution: {e}", exc_info=True)
        return {"error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    # 実行方法: uvicorn main:app --host 0.0.0.0 --port 8001 --reload
    # NOTE: The module name 'main' must match the Python script name (e.g., main.py)
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
