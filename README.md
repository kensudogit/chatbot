# チャットボット

## 機能
- 自然言語処理による対話
  - 表記ゆれの自動吸収
  - 曖昧な質問への適切な対応
  - 感情分析に基づく応答調整
  - エンティティ抽出による文脈理解
- ユーザー認証
- データベース統合
- パフォーマンスモニタリング
- キャッシュ機能
- 非同期処理
- 管理画面

## 必要条件
- Python 3.8以上
- SQLite（開発環境用）
- Redis（オプション）

## セットアップ手順

### 1. 環境の準備
```bash
# プロジェクトディレクトリに移動
cd chatbot/backend

# 仮想環境の作成
python -m venv venv

# 仮想環境の有効化
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. 必要なパッケージのインストール
```bash
# 基本的な依存パッケージのインストール
pip install -r requirements.txt

# 追加で必要なパッケージのインストール
pip install email-validator  # メールアドレス検証用
pip install sentencepiece   # 日本語トークナイザー用
```

### 3. データベースの初期化
```bash
# データベースの初期化（自動的に実行されます）
python main.py
```

### 4. サーバーの起動
```bash
# 開発サーバーの起動
python main.py
```

サーバーが起動すると、以下のエンドポイントが利用可能になります：
- API: http://localhost:8000
- APIドキュメント: http://localhost:8000/docs

### 5. 初期管理者アカウントの設定
```bash
# 管理者アカウントの作成
# Linux/Mac:
curl -X POST http://localhost:8000/api/setup/admin

# Windows (PowerShell):
Invoke-WebRequest -Method POST -Uri http://localhost:8000/api/setup/admin

# 作成される管理者アカウントの認証情報
メールアドレス: admin@example.com
パスワード: admin123
```

### 6. ログインとアクセストークンの取得
```bash
# アクセストークンの取得
# Linux/Mac:
curl -X POST http://localhost:8000/api/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin@example.com&password=admin123"

# Windows (PowerShell):
$body = @{
    username = "admin@example.com"
    password = "admin123"
}
Invoke-WebRequest -Method POST -Uri http://localhost:8000/api/token `
  -ContentType "application/x-www-form-urlencoded" `
  -Body $body

# レスポンス例
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1...",
  "token_type": "bearer"
}
```

## フロントエンドのセットアップ

### 1. 環境の準備
```bash
# プロジェクトディレクトリに移動
cd chatbot/frontend

# 依存パッケージのインストール
npm install
# または
yarn install
```

### 2. 開発サーバーの起動
```bash
# 開発サーバーの起動
npm run serve
# または
yarn serve
```

フロントエンドサーバーが起動すると、以下のURLでアクセス可能になります：
- ローカル: http://localhost:8081
- ネットワーク: http://[あなたのIPアドレス]:8081

## 注意事項
- フロントエンドは8081ポートで動作します
- バックエンドは8000ポートで動作します
- メッセージの送信時は、バックエンドサーバーが起動していることを確認してください
- シークレットモードを使用する場合は、メッセージの内容が暗号化されます
- ダークモードとライトモードは、UIの見た目を切り替えます

## トラブルシューティング
### よくある問題と解決方法

1. メッセージが表示されない場合
   - バックエンドサーバーが起動していることを確認
   - ブラウザの開発者ツールでエラーメッセージを確認
   - ログイン状態を確認

2. メッセージ送信時にエラーが発生する場合
   - バックエンドサーバーのログを確認
   - 認証トークンが有効か確認
   - ネットワーク接続を確認

3. フロントエンドサーバーが起動しない場合
   - ポート8081が他のプロセスで使用されていないか確認
   - 依存パッケージが正しくインストールされているか確認
   - Node.jsのバージョンを確認

4. バックエンドサーバーが起動しない場合
   - ポート8000が他のプロセスで使用されていないか確認
   - Pythonの仮想環境が有効になっているか確認
   - 必要なパッケージがすべてインストールされているか確認

### ログの確認方法

1. フロントエンドのログ
   - ブラウザの開発者ツール（F12）を開く
   - Consoleタブでエラーメッセージを確認
   - NetworkタブでAPIリクエストの状態を確認

2. バックエンドのログ
   - ターミナルでバックエンドサーバーの出力を確認
   - エラーメッセージや警告を確認
   - データベースの接続状態を確認
