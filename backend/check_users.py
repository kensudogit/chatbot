from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import User, Base
import os
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# データベース接続URL
DATABASE_URL = os.getenv('DATABASE_URL', 'mysql+pymysql://root:1qazXSW20708@localhost/chatbot?charset=utf8mb4')

# データベースエンジンの作成
engine = create_engine(DATABASE_URL)

# テーブルの作成（存在しない場合）
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)

def check_users():
    session = Session()
    try:
        # 全ユーザーの取得
        users = session.query(User).all()
        
        print("\n=== Users Table Contents ===")
        print(f"Total users: {len(users)}\n")
        
        for user in users:
            print(f"ID: {user.id}")
            print(f"Username: {user.username}")
            print(f"Email: {user.email}")
            print(f"Is Admin: {user.is_admin}")
            print(f"Created At: {user.created_at}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        session.close()

if __name__ == "__main__":
    check_users() 