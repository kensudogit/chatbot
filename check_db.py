from chatbot import app, db
from models import User, ChatHistory, Intent, ResponseTemplate

def check_database():
    with app.app_context():
        # テーブルの一覧を表示
        print("\n=== テーブル一覧 ===")
        for table in db.engine.table_names():
            print(f"- {table}")
        
        # 各テーブルのレコード数を表示
        print("\n=== レコード数 ===")
        print(f"User: {User.query.count()}件")
        print(f"ChatHistory: {ChatHistory.query.count()}件")
        print(f"Intent: {Intent.query.count()}件")
        print(f"ResponseTemplate: {ResponseTemplate.query.count()}件")
        
        # ユーザーテーブルの内容を表示
        print("\n=== ユーザー一覧 ===")
        users = User.query.all()
        for user in users:
            print(f"ID: {user.id}, ユーザー名: {user.username}, メール: {user.email}, 管理者: {user.is_admin}")

if __name__ == '__main__':
    check_database() 