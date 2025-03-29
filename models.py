"""
データベースモデル定義
SQLAlchemyを使用したORMモデル
- ユーザー管理
- チャット履歴
- 意図分類
- レスポンステンプレート
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from sqlalchemy import Index

# SQLAlchemyインスタンスの初期化
db = SQLAlchemy()

class User(UserMixin, db.Model):
    """
    ユーザーモデル
    - ユーザー認証情報の管理
    - チャット履歴との関連付け
    - 管理者権限の管理
    """
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    chat_history = db.relationship('ChatHistory', backref='user', lazy='select')
    is_admin = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f'<User {self.username}>'

class ChatHistory(db.Model):
    """
    チャット履歴モデル
    - ユーザーとの会話記録
    - 意図分類結果の保存
    - タイムスタンプ管理
    """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False, index=True)
    message = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    intent = db.Column(db.String(50), index=True)
    confidence = db.Column(db.Float, index=True)

class Intent(db.Model):
    """
    意図分類モデル
    - ユーザーの意図を定義
    - キーワードベースの分類
    - レスポンステンプレートとの関連付け
    """
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    description = db.Column(db.Text)
    keywords = db.Column(db.Text)  # JSON形式で保存
    response_template = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ResponseTemplate(db.Model):
    """
    レスポンステンプレートモデル
    - チャットボットの応答テンプレート
    - 変数置換機能
    - アクティブ/非アクティブ管理
    """
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    content = db.Column(db.Text, nullable=False)
    variables = db.Column(db.Text)  # JSON形式で保存
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# パフォーマンス最適化のための複合インデックス
Index('idx_chat_history_user_timestamp', ChatHistory.user_id, ChatHistory.timestamp.desc())
Index('idx_chat_history_intent_confidence', ChatHistory.intent, ChatHistory.confidence) 