import os
import uvicorn
import operator
from typing import TypedDict, Annotated, List, Union, Tuple

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM
# 修正: toolsのインポートをLangChainの推奨パスに変更
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException # <-- OutputParserExceptionをインポート

# LangGraph関連のインポート
from langgraph.graph import StateGraph, END

# --- 1. LLMとツールの設定 ---
# OllamaをLLMとして設定
# 注意: 実行前にOllamaサーバー (http://localhost:11434) が起動している必要があります
llm = OllamaLLM(model="llama3:8b-instruct-q4_K_M", base_url="http://localhost:11434")

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
    """Use this tool when information is not available on Wikipedia, 
    or when you need broader web search for recent news, events, or specific product information."""
    return duckduckgo_search.run(query)

tools = [search_wikipedia, search_web]


# --- 2. LangGraphの状態 (State) の定義 ---
# AgentActionとAgentFinishも含まれるため、Unionで型を拡張します。
MessageType = Union[BaseMessage, AgentAction, AgentFinish]

class AgentState(TypedDict):
    """
    LangGraph全体で共有される状態を定義します。
    ここでは、メッセージ履歴のみを保持します。
    """
    messages: Annotated[List[MessageType], operator.add]

# --- ヘルパー関数: メッセージ履歴をAgentAction/Observationのタプル形式に変換 (FIX) ---
def get_intermediate_steps(messages: List[MessageType]) -> List[Tuple[AgentAction, str]]:
    """
    LangGraphのメッセージ履歴から、AgentActionとToolMessageのペアを抽出し、
    LangChainのReActエージェントが期待する 'intermediate_steps' の形式に変換します。
    """
    filtered_history = [
        msg for msg in messages 
        if isinstance(msg, AgentAction) or isinstance(msg, ToolMessage)
    ]

    pairs = []
    # AgentActionとToolMessageが交互に来ることを想定してペアを作成
    for i in range(0, len(filtered_history), 2):
        if i + 1 < len(filtered_history):
            action = filtered_history[i]
            observation = filtered_history[i+1]
            
            if isinstance(action, AgentAction) and isinstance(observation, ToolMessage):
                 # AgentActionとObservation文字列のタプル (LangChainの要件)
                 pairs.append((action, observation.content)) 

    return pairs


# --- 3. プロンプトとRunnable Agentの定義 ---
# LangGraphでReActを実装するためには、新しいRunnable Agentの構築が必要です。
# これは、LLMの呼び出しを担うノードとして機能します。

# ReActプロンプトのテンプレート
# 文字列をPromptTemplateオブジェクトに変換します
prompt_template = PromptTemplate.from_template("""
You are an **extremely strict** AI assistant designed to act only as a reasoning engine. **DO NOT be conversational.**
Your only goal is to follow the ReAct format precisely to answer the Human's request.

**Crucial Rule:** If the input is a simple greeting or non-question (e.g., "hello", "thank you", or blank), immediately output a Final Answer without using a tool (e.g., 'Final Answer: Hello! How can I help you?').

Follow these steps to generate the best possible answer:

1. Thought
   - Reason step-by-step how to solve the question.
   - Determine which tool (search_wikipedia or search_web) is appropriate.
   - If a tool is necessary, output the Action and Action Input in strict JSON format.
   - If no tool is needed, or if the final answer has been reached, provide the final response prefixed with 'Final Answer:'.

2. Action
   - Only when necessary, call one of the available tools: {tool_names}.

3. Observation
   - The result of the tool call will be displayed here.

Available tools:
{tools}

Human: {input}
{agent_scratchpad}
""")

# LangChainのRunnable Agent（LCELベース）の構築
# これがLangGraphのノードとして機能します
runnable_agent = create_react_agent(llm, tools, prompt_template)


# --- 4. グラフのノード関数の定義 ---

def run_agent(state: AgentState):
    """LLMを呼び出し、次のActionまたはFinal Answerを決定します。"""
    print("--- ノード: run_agent (LLM呼び出し) ---")
    
    # --- FIX: 'intermediate_steps' Key Errorを解決する ---
    # 1. メッセージ履歴からAction/Observationのタプルリストを抽出
    intermediate_steps = get_intermediate_steps(state["messages"])
    
    # 2. ReActプロンプトの {input} に渡すため、オリジナルのユーザー入力を抽出
    original_input_message = next(
        (msg for msg in state["messages"] if isinstance(msg, HumanMessage)), 
        state["messages"][0] if state["messages"] else HumanMessage(content="")
    )

    try:
        # 3. Runnable Agentを実行する際、'intermediate_steps' キーに履歴を渡す
        result = runnable_agent.invoke({
            "input": original_input_message.content, 
            "tools": tools, 
            "intermediate_steps": intermediate_steps # <--- キー名を修正
        })
    except OutputParserException as e:
        # --- FIX: OutputParserExceptionをキャッチし、LLMに修正を促すメッセージを返す ---
        print(f"--- PARSING ERROR CAUGHT. Returning to agent. --- Error: {e}")
        
        # LLMの不正な出力をToolMessageとしてラップし、LLMにフィードバック
        # NOTE: e.llm_outputはOutputParserExceptionから取得可能
        llm_output = getattr(e, "llm_output", "N/A (Output missing)")
        
        error_message = (
            f"PARSING ERROR: Your previous response could not be parsed. "
            f"The failed output was: '{llm_output}'. "
            f"You MUST strictly follow the ReAct format (Thought, Action/Final Answer). "
            f"Please try again."
        )
        return {"messages": [HumanMessage(content=error_message)]}

    
    # AgentAction または AgentFinish をメッセージに追加
    if isinstance(result, list):
        # ツールを呼び出す場合 (AgentActionのリストが返る)
        return {"messages": result}
    elif isinstance(result, AgentAction):
        # AgentActionが直接返る場合
        return {"messages": [result]}
    elif isinstance(result, AgentFinish):
        # 最終的な回答が返る場合
        return {"messages": [result]}
    else:
        # 予期せぬ形式の場合、エラーメッセージとして返す
        # HumanMessage(content="") を使って状態を更新し、ENDに導くか、エラーとして表示
        return {"messages": [HumanMessage(content=f"Error: Unexpected output type from agent: {type(result)}")]}


def execute_tools(state: AgentState):
    """LLMが選択したツールを実行し、結果を返します。"""
    print("--- ノード: execute_tools (ツール実行) ---")
    
    # 最新のActionを取得
    action = state["messages"][-1]
    
    tool_results = []
    
    if not isinstance(action, AgentAction):
        # AgentActionでなければ、ツール実行をスキップ
        print(f"Warning: Expected AgentAction, but received {type(action)}. Skipping tool execution.")
        # この場合、状態を更新しない (Noneを返す)
        return None 

    # ツールをディスパッチして実行
    tool_map = {tool.name: tool for tool in tools}
    
    # Actionを実行
    if action.tool in tool_map:
        tool_function = tool_map[action.tool]
        try:
            # ツールの実行 (Observationを取得)
            observation = tool_function.run(action.tool_input)
            
            # 結果をToolMessageとしてラップし、次の思考ステップに戻す
            tool_message = ToolMessage(
                content=str(observation), 
                name=action.tool,
                # create_react_agentを使用しているため、tool_call_idは省略します
            )
            tool_results.append(tool_message)
        except Exception as e:
            error_message = f"Tool execution failed for {action.tool}: {e}"
            tool_results.append(ToolMessage(content=error_message, name=action.tool))
    else:
        error_message = f"Unknown tool: {action.tool}. Available tools: {list(tool_map.keys())}"
        tool_results.append(ToolMessage(content=error_message, name="error"))
        
    return {"messages": tool_results}


# --- 5. 条件付きエッジのロジック ---

def should_continue(state: AgentState):
    """
    LLMの出力がActionかFinal Answerかを判断し、次のノードを決定します。
    """
    last_message = state["messages"][-1]

    if isinstance(last_message, AgentFinish):
        # AgentFinish（Final Answer）の場合、終了 (END)
        print("--- 条件: AgentFinish. グラフを終了します。---")
        return "end"
    
    # それ以外の場合（AgentActionの場合）、ツール実行へ
    print("--- 条件: AgentActionを検出。ツール実行へ遷移します。---")
    return "continue"


# --- 6. グラフの構築とコンパイル ---

# グラフを構築
workflow = StateGraph(AgentState)

# ノードを追加
workflow.add_node("agent", run_agent)
workflow.add_node("tools", execute_tools)

# エントリポイントを設定
workflow.set_entry_point("agent")

# 条件付きエッジを設定: LLMの結果によって次を決定
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools", # ツールを使う場合
        "end": END           # 最終回答の場合
    },
)

# 標準のエッジを設定: ツール実行後は必ずエージェント（LLM）に戻る
workflow.add_edge("tools", "agent")

# グラフをコンパイル
app_graph = workflow.compile()


# --- 7. FastAPIの定義とエンドポイント ---

# FastAPIのインスタンスを作成
app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat_with_llama3(request: PromptRequest):
    """
    LangGraphベースのエージェントにプロンプトを送信し、応答を取得します。
    """
    try:
        # ユーザーのプロンプトをHumanMessageとしてLangGraphの状態の初期値に設定
        initial_state = {"messages": [HumanMessage(content=request.prompt)]}
        
        # グラフを実行
        result = app_graph.invoke(initial_state)

        # 最終的なAgentFinishメッセージから回答テキストを抽出
        final_message = result["messages"][-1]
        
        if isinstance(final_message, AgentFinish):
            response_text = final_message.return_values["output"]
        else:
            # 予期せぬ終了の場合（デバッグ用）
            response_text = f"Error or unexpected output format: {final_message}"

        return {"response": response_text}
        
    except Exception as e:
        import traceback
        # 実行ログを確認しやすくするため、エラーメッセージにトレースバックを含めます
        return {"error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    # 実行方法: uvicorn main:app --host 0.0.0.0 --port 8001 --reload
    # NOTE: The module name 'main' must match the Python script name (e.g., main.py)
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
