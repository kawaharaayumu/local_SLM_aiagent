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
    'per_page': 10
}

response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    items = response.json()
    print(f"取得件数: {len(items)}件")
    for item in items:
        # タイトルとURLを表示
        print(f"Title: {item['title']}")
        print(f"URL:   {item['url']}")
        print("-" * 20)
else:
    print(f"Error: {response.status_code}")