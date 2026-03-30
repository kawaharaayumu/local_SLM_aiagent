# AIagent
## Local LLM Multi-Tool Agent (LangGraph + FastAPI)
このプロジェクトは、Ollama で動作するローカルLLMを LangGraph で制御し、Web検索やWikipediaからの情報取得を可能にしたAIエージェントのバックエンドAPIです。

## 🚀 主な機能
- ローカル推論: Ollama (qwen3.5:9b 推奨) を使用したプライバシー重視の動作。
- 状態管理 (LangGraph): 会話の文脈を保持し、必要に応じてツールを自律的に呼び出し。
- マルチツール: DuckDuckGoによるWeb検索とWikipedia検索を統合。
- ハイブリッド・ロジック: 最終回答はユーザーの言語に合わせつつ、内部推論やツール利用には英語や中国語などの広範なリソースを活用。

## 準備するもの
1. Ollama のセットアップ
Ollamaをインストールし、モデルを準備します。
`` ollama pull qwen3.5:9b ``

2. ライブラリのインストール
`` pip install -r requirements.txt ``

## 実行方法
`` uvicorn main:app --host 0.0.0.0 --port 8001 --reload ``

## 📡 API リファレンス
### エージェントへのリクエスト
- URL:``/agent``
- Method: ``POST``
- Payload:
``{
  "prompt": "日本で一番高い山は？",
  "thread_id": "session_001"
}``

### レスポンス例
`` {
  "response": "日本で一番高い山は**富士山**です。\n\n### 富士山の概要\n- **標高**：約3,776m\n- **位置**：静岡県と山梨県の境界に位置\n- **特徴**：日本を代表する活火山で、美しい山容と歴史的な価値を持って日本人に愛されています\n\n### 他の有名な高い山\n1位：富士山（約3,776m）\n2位：北岳（約3,193m）\n3位：奥穂高岳（約3,190m）\n\nこれら3つを「日本三岳」と呼ぶことがよくあります。"
} ``


# frontend
AIエージェントとやり取りするためのフロントエンド。