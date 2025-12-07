import requests
import json
import sys
import os
from dotenv import load_dotenv # 💡 この行を追加

# .envファイルをロードする
# スクリプトと同じディレクトリにある .env ファイルから環境変数を読み込む
load_dotenv() # 💡 この行を追加

# 環境変数からAPI情報を取得
# 💡 環境変数として定義したキー名を使用
api_host = os.getenv("API_HOST")
api_port = os.getenv("API_PORT")
api_path = os.getenv("API_PATH")

# 環境変数が設定されているか確認
if not all([api_host, api_port, api_path]):
    print("エラー: .envファイルにAPI_HOST, API_PORT, または API_PATHが設定されていません。")
    sys.exit(1)

# APIエンドポイントのURLを構築
# f-stringを使用して変数を埋め込む
url = f"http://{api_host}:{api_port}{api_path}" # 💡 URLの構築方法を変更

# コマンドライン引数からプロンプトを取得
# スクリプト名を除いた引数を結合してプロンプトにする
if len(sys.argv) > 1:
    prompt_text = " ".join(sys.argv[1:])
else:
    # 引数がない場合はエラーメッセージを表示して終了
    print("使用方法: python dev.py <質問内容>")
    sys.exit(1)


# 送信するデータ（プロンプト）
# APIの仕様に合わせてPromptRequestの形式でデータを構築
# この場合は {"prompt": "ここに質問内容"} という形式
data = {
    "prompt": prompt_text
}

# ヘッダーにContent-Typeを指定
# JSONデータを送ることをAPIに伝えるため
headers = {
    "Content-Type": "application/json"
}

try:
    # POSTリクエストを送信
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    # 成功した場合のステータスコードは200
    if response.status_code == 200:
        # レスポンスをJSON形式で受け取る
        result = response.json()
        print("APIからの応答:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 応答から「response」キーの値を取得
        print("\n回答:")
        print(result.get("response"))
        
    else:
        # エラーが発生した場合
        print(f"エラー: ステータスコード {response.status_code}")
        print(f"レスポンス: {response.text}")

except requests.exceptions.RequestException as e:
    # リクエスト自体が失敗した場合（例: サーバーが起動していない）
    print(f"リクエストエラーが発生しました: {e}")
