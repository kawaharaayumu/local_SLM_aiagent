from langchain_community.tools import DuckDuckGoSearchRun

# DuckDuckGoウェブ検索ツールを定義
duckduckgo_search = DuckDuckGoSearchRun()

def search_web(query: str) -> str:
    """Search the web using DuckDuckGo."""
    return duckduckgo_search.run(query)

# ツールとして定義された関数を使用
query = "Japan"
result_google = search_web(query)
print(f"Simple Serch Result: {result_google}")
# 1. Google検索にリダイレクトする例
# 検索結果はDuckDuckGoを経由せず、Googleの結果を直接取得します
google_query = "!g "+ query
result_google = search_web(google_query)
print(f"Google Search Result (via !g): {result_google}")

# 2. Wikipedia (日本語) サイト内検索の例
# Wikipediaの「LangChain」に関する記事を直接検索します
wikipedia_query = "!wj "+ query
result_wikipedia = search_web(wikipedia_query)
print(f"Wikipedia Search Result (via !wj): {result_wikipedia}")

# 3. YouTube検索の例
# YouTubeで「LangGraph チュートリアル」を検索します
youtube_query = "!yt " + query
result_youtube = search_web(youtube_query)
print(f"YouTube Search Result (via !yt): {result_youtube}")