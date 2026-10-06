# Chatbot Platform — FastAPI / Authentication / Conversation History

> **AI Application Foundation** — FastAPIをベースに、認証、チャットAPI、会話履歴、ヘルスチェック、パフォーマンス監視をまとめたバックエンド基盤です。
>
> **Stack:** Python 3.11 · FastAPI · MySQL · JWT · REST API

## Portfolio Position

このリポジトリは、LLMそのものよりも **AIチャットサービスを運用するためのアプリケーション基盤** に焦点を置いています。認証・履歴・API・監視を分離して実装し、より高度なAI Agent / RAGシステムへ発展させるための基礎構成を示します。

## Architecture

```text
Client
  │
  ├── Authentication
  │      └── JWT / session management
  │
  └── Chat REST API
         ├── conversation processing
         ├── history management
         └── monitoring / health
                    │
                    ▼
                  MySQL
```

## 主な機能

- ユーザー登録・ログイン・ログアウト
- JWT Bearer認証
- チャットメッセージAPI
- 会話履歴管理
- 管理者アカウント管理
- ヘルスチェック
- パフォーマンスモニタリング
- APIエラーハンドリング

## 必要条件
- Python 3.11以上
- MySQL 8.0以上
- pip

## 開発環境のセットアップ

### 1. Python環境のセットアップ
```bash
# 仮想環境の作成
python -m venv venv311

# 仮想環境のアクティベート
# Windows
.\venv311\Scripts\activate
# Linux/Mac
source venv311/bin/activate

# pipのアップグレード
python -m pip install --upgrade pip

# 依存関係のインストール
pip install -r requirements.txt
```

### 2. データベースのセットアップ
1. MySQLサーバーを起動
2. データベースとユーザーを作成
```sql
CREATE DATABASE chatbot CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'chatbot_user'@'localhost' IDENTIFIED BY '<your-password>';
GRANT ALL PRIVILEGES ON chatbot.* TO 'chatbot_user'@'localhost';
FLUSH PRIVILEGES;
```

### 3. 環境変数の設定
`.env`ファイルを作成し、以下の内容を設定：
```env
SECRET_KEY=<generate-a-strong-random-secret>
JWT_SECRET_KEY=<generate-a-strong-random-secret>
DATABASE_URL=mysql://chatbot_user:<your-password>@localhost/chatbot?charset=utf8mb4
PORT=8000
```

### 4. アプリケーションの起動
```bash
# 仮想環境がアクティベートされていることを確認
# Windows
.\venv311\Scripts\activate
# Linux/Mac
source venv311/bin/activate

# アプリケーションの起動
python main.py
```

## デフォルト管理者アカウント
- ユーザー名: admin
- パスワード: admin123
- メールアドレス: admin@example.com

## APIエンドポイント
- `POST /register`: 新規ユーザー登録
- `POST /login`: ユーザーログイン
- `POST /logout`: ログアウト
- `POST /chat`: チャットメッセージの送信
- `GET /chat/history`: チャット履歴の取得
- `GET /health`: ヘルスチェック
- `POST /create-admin`: 管理者アカウントの作成

## 開発時の注意事項
- デバッグモードが有効になっています（`debug=True`）
- デフォルトポートは8000です
- セキュリティ関連の設定は開発用に簡略化されています
- 本番環境では適切なセキュリティ設定が必要です

## エラーハンドリング
### 認証エラー
- トークンの有効期限切れ（401エラー）
  - 自動的にログイン画面にリダイレクト
  - ローカルストレージのトークンが削除される
- 認証失敗（401エラー）
  - ログイン画面にリダイレクト
  - トークンが無効な場合は削除

### サーバーエラー
- 500エラー（Internal Server Error）
  - サーバー側でエラーが発生
  - エラーメッセージが表示される
  - しばらく時間をおいて再度試行

### クライアントエラー
- 404エラー（Not Found）
  - リクエストしたリソースが見つからない
  - APIエンドポイントのパスを確認

## トークン管理
- アクセストークンの有効期限：7日間
- トークンの保存場所：ローカルストレージ
- トークンの自動更新：なし（期限切れ時は再ログインが必要）
- セキュリティ対策：
  - トークンはBearer認証で送信
  - セッション管理による追加のセキュリティ
  - ログアウト時にトークンを無効化

## トラブルシューティング
1. データベース接続エラー
   - MySQLサーバーが起動していることを確認
   - データベースとユーザーが正しく作成されていることを確認
   - 環境変数の設定を確認

2. 依存関係のインストールエラー
   - Python 3.11を使用していることを確認
   - 仮想環境が正しくアクティベートされていることを確認
   - pipを最新バージョンにアップグレード

3. アプリケーション起動エラー
   - 必要なポートが使用可能であることを確認
   - 環境変数が正しく設定されていることを確認
   - ログを確認して具体的なエラーメッセージを確認

4. 認証エラー
   - トークンが有効期限内であることを確認
   - ログイン情報が正しいことを確認
   - ブラウザのローカルストレージを確認

5. API通信エラー
   - バックエンドサーバーが起動していることを確認
   - APIエンドポイントのパスが正しいことを確認
   - ネットワーク接続を確認
   - CORS設定を確認 

### Security note

`SECRET_KEY` is required at runtime. Do not commit real credentials or secrets; inject them through environment variables or your deployment platform's secret manager.
