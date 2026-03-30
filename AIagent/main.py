import uvicorn
import logging
import sys
import time
from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel

# LangChain / LangGraph 関連のインポート（重複を整理）
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# --- 1. ロガーのシンプル化 ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # DEBUGからINFOに変更し、不要な出力を抑制

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    # ログの見た目をシンプルに
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- 2. ツールの定義 ---
wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=700)
wikipedia_tool_base = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)
duckduckgo_search = DuckDuckGoSearchRun()

@tool
def search_wikipedia(query: str) -> str:
    """Use this tool for searching content that has a dedicated Wikipedia article."""
    return wikipedia_tool_base.run(query)

@tool
def search_web(query: str) -> str:
    """Search the web using DuckDuckGo."""
    return duckduckgo_search.run(query)

# tool
# tools = [search_web, search_wikipedia]
tools = [search_web]
tool_node = ToolNode(tools)

# --- 3. モデルの定義 ---
model = ChatOllama(model="qwen3.5:4b", base_url="http://localhost:11434").bind_tools(tools)

# --- 4. グラフの構築 (API外で1回だけ実行) ---
def should_continue(state: MessagesState) -> Literal["tools", END]:
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "tools"
    return END

def call_model(state: MessagesState):
    response = model.invoke(state['messages'])
    return {"messages": [response]}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# コンパイルをグローバルで1度だけ行うことでパフォーマンス向上
checkpointer = MemorySaver()
lang_app = workflow.compile(checkpointer=checkpointer)

# --- 5. FastAPI アプリケーション ---
app = FastAPI(title="Local LLM Agent API")

class PromptRequest(BaseModel):
    prompt: str
    # 修正点: thread_idを任意で受け取れるようにし、デフォルト値を設定
    # これにより、同じthread_idを使えば会話の記憶が引き継がれます
    thread_id: str = "default_session"

def log_message(msg):
    """コンソール出力を分かりやすくフォーマットするヘルパー関数"""
    if isinstance(msg, HumanMessage):
        logger.info(f"\n👤 [User]: {msg.content}")
    elif isinstance(msg, AIMessage):
        if msg.tool_calls:
            tools_str = ", ".join([f"{tc['name']}({tc['args']})" for tc in msg.tool_calls])
            logger.info(f"🤖 [Agent]: ツールを実行します -> 🛠️ {tools_str}")
        elif msg.content:
            # ターミナルが埋まらないように、思考プロセスの長文は省略して表示
            content_preview = msg.content.replace('\n', ' ')
            if len(content_preview) > 100:
                content_preview = content_preview[:100] + "..."
            logger.info(f"🤖 [Agent]: 💬 {content_preview}")
    elif isinstance(msg, ToolMessage):
        logger.info(f"🛠️ [Tool]: ✅ '{msg.name}' の検索が完了しました。")

@app.post("/agent")
async def agent_by_langgraph(request: PromptRequest):
    logger.info("-" * 40)
    logger.info(f"🚀 新しいリクエストを受信 (Thread: {request.thread_id})")
    
    config = {"configurable": {"thread_id": request.thread_id}}
    final_response_text = "応答を生成できませんでした。"
    
    try:
        inputs = {"messages": [HumanMessage(content=request.prompt)]}
        
        # ログ出力用：既に処理したメッセージのIDを保持
        processed_ids = set()
        
        for event in lang_app.stream(inputs, config, stream_mode="values"):
            messages = event.get("messages", [])
            if not messages:
                continue
                
            last_message = messages[-1]
            
            # 同じメッセージを何度もログ出力しないための制御
            if id(last_message) not in processed_ids:
                log_message(last_message)
                processed_ids.add(id(last_message))
                
            # 最終的なテキスト応答を更新
            if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                if last_message.content:
                    final_response_text = last_message.content

        logger.info("✨ 処理完了")
        return {"response": final_response_text}

    except Exception as e:
        logger.error(f"❌ エラーが発生しました: {e}", exc_info=True)
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)