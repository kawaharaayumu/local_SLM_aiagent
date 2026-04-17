import requests
import os
from dotenv import load_dotenv

load_dotenv()
# ここに取得したアクセストークンを設定
ACCESS_TOKEN = os.getenv('QIITA_ACCESS_TOKEN')

headers = {
    'Authorization': f'Bearer {ACCESS_TOKEN}'
}

# 記事一覧を取得するAPIエンドポイント
url = 'https://qiita.com/api/v2/items'

# パラメータ設定（例：1ページあたり10件取得）
params = {
    'page': 1,
    'per_page': 10,
    'query': 'tag:AIエージェント stocks:>10' # Pythonタグで10LGTM以上の記事
}

response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    items = response.json()
    print(f"取得件数: {len(items)}件")
    for item in items:
        # タイトルとURLを表示
        print(f"Title: {item['title']}")
        print(f"URL:   {item['url']}")
        print(f"ID: {item['id']}")
        print("-" * 20)
else:
    print(f"Error: {response.status_code}")
    


## ここからはID検索
ITEM_ID = 'ebbbe74649eb441a34db' 

url = f'https://qiita.com/api/v2/items/{ITEM_ID}'
headers = {'Authorization': f'Bearer {ACCESS_TOKEN}'}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    item = response.json()
    print(f"【タイトル】: {item['title']}")
    print(f"【投稿者】  : {item['user']['id']}")
    print(f"【LGTM数】  : {item['likes_count']}")
    print("-" * 30)
    # body に Markdown形式の本文が入っています
    print("【本文（冒頭）】:")
    print(item['body'][:300] + "...") # 長いので最初の300文字だけ表示
else:
    print(f"Error: {response.status_code}")