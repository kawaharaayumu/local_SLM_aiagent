import requests
import json

def getWikiData(url, params):
    # Wikipedia APIにはUser-Agentヘッダーが必須です
    # 連絡先メールアドレスなどを入れるのが推奨されていますが、適当な文字列でも動きます
    headers = {
        'User-Agent': 'MyTestBot/1.0 (contact@example.com)'
    }
    
    res = requests.get(url, params=params, headers=headers)
    
    # 念のため、レスポンスが正常かチェック
    res.raise_for_status()
    
    return res.json()

url = "https://ja.wikipedia.org/w/api.php"
params = { 
    "action" : "query",
    "titles" : "Python",
    "prop"   : "extracts", # 内容を取得したい場合は追加
    "exintro": True,       # 導入部のみ
    "explaintext": True,   # プレーンテキストで取得
    "format" : "json"
}

print(getWikiData(url, params))