# チャットボットアプリケーション

## 概要
このプロジェクトは、FastAPIを使用したRESTful APIサーバーをベースとしたチャットボットアプリケーションです。
- ユーザー認証
- チャット機能
- 会話履歴管理
- パフォーマンスモニタリング

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
CREATE USER 'chatbot_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON chatbot.* TO 'chatbot_user'@'localhost';
FLUSH PRIVILEGES;
```

### 3. 環境変数の設定
`.env`ファイルを作成し、以下の内容を設定：
```env
SECRET_KEY=your-secret-key
JWT_SECRET_KEY=your-jwt-secret-key
DATABASE_URL=mysql://chatbot_user:your_password@localhost/chatbot?charset=utf8mb4
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