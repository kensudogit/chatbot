"""
データベースモデル定義
- SQLAlchemyを使用したORMモデル
- ユーザー、チャット、メッセージの管理
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, Boolean, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

# データベース設定
SQLALCHEMY_DATABASE_URL = "sqlite:///chatbot.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=True
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# パスワードハッシュ化
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT設定
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7日間

class User(Base):
    """
    ユーザーモデル
    - ユーザー情報の管理
    - 認証情報の保存
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    email = Column(String(100), unique=True, index=True)
    hashed_password = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    last_login = Column(DateTime)
    profile = relationship("UserProfile", back_populates="user", uselist=False)
    chats = relationship("Chat", back_populates="user")
    activity_logs = relationship("UserActivityLog", back_populates="user")

class UserProfile(Base):
    """
    ユーザープロファイルモデル
    - ユーザーの追加情報
    - 設定と環境設定
    """
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    full_name = Column(String(100))
    bio = Column(Text)
    avatar_url = Column(String(255))
    preferences = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = relationship("User", back_populates="profile")

class Chat(Base):
    """
    チャットモデル
    - チャットセッションの管理
    - メッセージのグループ化
    """
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String(200))
    is_secret = Column(Boolean, default=False)  # シークレットモードフラグ
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    messages = relationship("Message", back_populates="chat")
    user = relationship("User", back_populates="chats")
    tags = relationship("ChatTag", back_populates="chat")

class Message(Base):
    """
    メッセージモデル
    - チャットメッセージの保存
    - 感情分析結果の保存
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("chats.id"))
    role = Column(String(20))
    content = Column(Text)
    sentiment = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    chat = relationship("Chat", back_populates="messages")

class ChatTag(Base):
    """
    チャットタグモデル
    - チャットの分類とフィルタリング
    - 検索の最適化
    """
    __tablename__ = "chat_tags"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("chats.id"))
    tag = Column(String(50))
    chat = relationship("Chat", back_populates="tags")

class UserActivityLog(Base):
    """
    ユーザー活動ログモデル
    - ユーザーの行動追跡
    - システム分析とモニタリング
    """
    __tablename__ = "user_activity_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    activity_type = Column(String(50))
    description = Column(Text)
    details = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="activity_logs")

class SystemMetrics(Base):
    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, index=True)
    metric_type = Column(String(50))
    value = Column(Float)
    details = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserSession(Base):
    """
    ユーザーセッションモデル
    - アクティブなセッションの管理
    - ログアウト状態の追跡
    """
    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    token = Column(String(255), unique=True, index=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    last_activity = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", backref="sessions")

# データベースの初期化
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=7)  # トークンの有効期限を7日に延長
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[str]:
    """
    トークンを検証し、ユーザーのメールアドレスを取得する
    
    Args:
        token: JWTトークン
        
    Returns:
        str: ユーザーのメールアドレス
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        return email
    except JWTError:
        return None

def get_current_user_from_token(token: str, db) -> Optional[User]:
    """
    トークンからユーザーを取得する
    
    Args:
        token: JWTトークン
        db: データベースセッション
        
    Returns:
        User: ユーザーオブジェクト
    """
    email = verify_token(token)
    if not email:
        return None
        
    user = db.query(User).filter(User.email == email).first()
    return user if user and user.is_active else None 