import logging
from database import SessionLocal, User, Chat, Message, UserActivityLog, SystemMetrics
import sqlite3

# SQLAlchemyのログを無効化
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)

def check_database():
    try:
        # データベースに接続
        conn = sqlite3.connect('chatbot.db')
        cursor = conn.cursor()

        # テーブル一覧を取得
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("\nテーブル一覧:")
        for table in tables:
            print(f"- {table[0]}")

        # messagesテーブルの内容を確認
        if ('messages',) in tables:
            print("\nメッセージ一覧:")
            cursor.execute("SELECT id, role, content, created_at FROM messages ORDER BY created_at DESC LIMIT 5;")
            messages = cursor.fetchall()
            for msg in messages:
                print(f"\nID: {msg[0]}")
                print(f"Role: {msg[1]}")
                print(f"Content: {msg[2]}")
                print(f"Created at: {msg[3]}")

        conn.close()
    except sqlite3.Error as e:
        print(f"データベースエラー: {e}")

if __name__ == "__main__":
    check_database() 