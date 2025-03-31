from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# rootユーザーでの接続（データベース作成用）
ROOT_URL = "mysql+pymysql://root:1qazXSW20708@127.0.0.1?charset=utf8mb4"

def setup_database():
    # rootユーザーで接続
    root_engine = create_engine(ROOT_URL)
    
    try:
        with root_engine.connect() as conn:
            # データベースの作成
            conn.execute(text("CREATE DATABASE IF NOT EXISTS chatbot CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
            
            # 既存のユーザーを削除（存在する場合）
            conn.execute(text("DROP USER IF EXISTS 'chatbot_user'@'localhost'"))
            
            # ユーザーの作成と権限付与
            conn.execute(text("CREATE USER 'chatbot_user'@'localhost' IDENTIFIED BY '1qazXSW20708'"))
            conn.execute(text("GRANT ALL PRIVILEGES ON chatbot.* TO 'chatbot_user'@'localhost'"))
            conn.execute(text("FLUSH PRIVILEGES"))
            
            print("Database and user setup completed successfully!")
            
    except Exception as e:
        print(f"Error during setup: {str(e)}")
        raise
    finally:
        root_engine.dispose()

if __name__ == "__main__":
    setup_database() 