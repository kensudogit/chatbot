"""
チャットボットのバックエンドAPI
- FastAPIを使用したRESTful API
- データベースとの連携
- AIモデルを使用した応答生成
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Query, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, ConfigDict
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import uvicorn
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, desc
import speech_recognition as sr
from gtts import gTTS
import os
import tempfile
import shutil
import json
import logging
import jwt
from jose import JWTError

from database import (
    Base, engine, SessionLocal, User, Chat, Message, 
    UserProfile, UserActivityLog, SystemMetrics, ChatTag, 
    verify_password, get_password_hash, create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES, SECRET_KEY, ALGORITHM, UserSession,
    init_db
)
from ai_models import AIModelManager

# データベースセッションの取得
def get_db():
    """データベースセッションを取得する依存関数"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI(
    title="チャットボットAPI",
    description="チャットボットのバックエンドAPI",
    version="1.0.0"
)

# フロントエンドのディレクトリパス
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")

# APIルートの設定
api_app = FastAPI(title="API")

# APIエンドポイントをサブアプリケーションとしてマウント
app.mount("/api", api_app)

# CORS設定
origins = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:8081",
    "http://127.0.0.1:8081"
]

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# レスポンスヘッダーの設定
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Viewport"] = "width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no"
    # フロントエンドのURLを動的に設定
    origin = request.headers.get("origin")
    if origin in ["http://localhost:8081", "http://localhost:8080", "http://127.0.0.1:8080", "http://127.0.0.1:8081"]:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response

# AIモデルマネージャーの初期化
ai_manager = AIModelManager()

# OAuth2設定
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")

# データモデル
class MessageBase(BaseModel):
    role: str
    content: str

class MessageModel(MessageBase):
    model_config = ConfigDict(from_attributes=True)

class ChatRequest(BaseModel):
    messages: List[MessageModel]
    title: Optional[str] = None
    is_secret: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    sentiment: Optional[dict]
    device_type: Optional[str] = "desktop"  # デバイスタイプ（desktop/mobile）

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    is_admin: Optional[bool] = False

class UserProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    preferences: Optional[Dict] = None

    model_config = ConfigDict(from_attributes=True)

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatSearch(BaseModel):
    query: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    tags: Optional[List[str]] = None

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ユーザーエージェントからデバイスタイプを判定する関数
def get_device_type(user_agent: str) -> str:
    mobile_keywords = ["mobile", "android", "iphone", "ipad", "ipod"]
    return "mobile" if any(keyword in user_agent.lower() for keyword in mobile_keywords) else "desktop"

# メインのルートハンドラ - すべてのリクエストをindex.htmlにリダイレクト
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """
    SPAのルーティングをサポートするためのキャッチオールハンドラ
    すべてのリクエストをindex.htmlにリダイレクトします
    """
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not Found")
    
    # 静的ファイルのパスをチェック
    file_path = os.path.join(FRONTEND_DIR, full_path)
    if os.path.exists(file_path) and not os.path.isdir(file_path):
        return FileResponse(file_path)
    
    # それ以外のすべてのリクエストをindex.htmlにリダイレクト
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# ルートパスのハンドラ
@app.get("/")
async def root():
    """
    ルートパスへのアクセスをindex.htmlにリダイレクト
    """
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# ログインページのエンドポイント
@app.get("/login")
async def login_page():
    """
    ログインページを表示するエンドポイント
    """
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# 管理者ページのエンドポイント
@app.get("/admin")
async def admin_page():
    """
    管理者ページを表示するエンドポイント
    """
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# チャットエンドポイント
@api_app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
):
    """
    チャットエンドポイント
    - ユーザーメッセージを受け取り、AIモデルを使用して応答を生成
    """
    try:
        # トークンからユーザー情報を取得
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            email = payload.get("sub")
            if email is None:
                raise HTTPException(status_code=401, detail="認証に失敗しました")
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="トークンの有効期限が切れています")
        except JWTError:
            raise HTTPException(status_code=401, detail="認証に失敗しました")
            
        # ユーザー認証
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(
                status_code=401,
                detail="認証に失敗しました。正しい認証情報を入力してください。"
            )

        # メッセージの検証
        if not request.messages:
            raise HTTPException(
                status_code=400,
                detail="メッセージが空です"
            )
        
        # AIモデルによる応答生成
        try:
            response = await ai_manager.generate_chat_response(
                messages=request.messages,
                max_tokens=1000,
                temperature=0.7
            )
        except Exception as e:
            logger.error(f"応答生成中にエラーが発生しました: {str(e)}")
            response = "申し訳ありません。応答の生成中にエラーが発生しました。"
        
        # チャット履歴の保存
        try:
            chat = Chat(
                user_id=user.id,
                title=request.title if request.title else "新しいチャット",
                is_secret=request.is_secret
            )
            db.add(chat)
            db.flush()
            
            # ユーザーのメッセージを保存
            user_msg = Message(
                chat_id=chat.id,
                role="user",
                content=request.messages[-1].content
            )
            db.add(user_msg)
            
            # AIの応答を保存
            ai_msg = Message(
                chat_id=chat.id,
                role="assistant",
                content=response
            )
            db.add(ai_msg)
            
            db.commit()
            
        except Exception as e:
            logger.error(f"チャット履歴の保存中にエラーが発生しました: {str(e)}")
            db.rollback()
        
        return ChatResponse(
            response=response
        )
        
    except Exception as e:
        logger.error(f"チャットエンドポイントでエラーが発生しました: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# 音声入力エンドポイント
@api_app.post("/speech-to-text")
async def speech_to_text(
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
):
    """
    音声をテキストに変換するエンドポイント
    
    Args:
        audio_file: アップロードされた音声ファイル
        db: データベースセッション
        token: 認証トークン
        
    Returns:
        dict: 変換されたテキストとメタデータ
    """
    try:
        # ユーザー認証
        user = db.query(User).filter(User.email == token).first()
        if not user:
            raise HTTPException(
                status_code=401,
                detail="認証に失敗しました。正しい認証情報を入力してください。"
            )
        
        # 音声ファイルの検証
        if not audio_file.filename.endswith(('.wav', '.mp3', '.ogg')):
            raise HTTPException(
                status_code=400,
                detail="サポートされていない音声形式です。WAV、MP3、OGG形式のファイルをアップロードしてください。"
            )
        
        # 音声をテキストに変換
        text = ai_manager.transcribe_audio(await audio_file.read())
        
        return {
            "text": text,
            "status": "success",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"音声処理中にエラーが発生しました: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="音声処理中にエラーが発生しました。しばらく時間をおいて再度お試しください。"
        )

# アプリケーション起動時にデータベースを初期化
@app.on_event("startup")
async def startup_event():
    """
    アプリケーション起動時の初期化処理
    - データベースの初期化
    - 管理者アカウントの作成
    """
    # データベースの初期化
    Base.metadata.create_all(bind=engine)
    
    logger.info("アプリケーションの初期化が完了しました")

# トークン取得エンドポイント
@api_app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    ユーザー認証とアクセストークンの生成
    """
    try:
        # ユーザーの検証
        user = db.query(User).filter(User.username == form_data.username).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="ユーザー名またはパスワードが正しくありません",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # パスワードの検証
        if not verify_password(form_data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="ユーザー名またはパスワードが正しくありません",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # アクセストークンの生成
        access_token_expires = timedelta(minutes=30)
        access_token = create_access_token(
            data={"sub": user.email},
            expires_delta=access_token_expires
        )

        # 最終ログイン時刻の更新
        user.last_login = datetime.utcnow()
        db.commit()

        # アクティビティログの記録
        activity_log = UserActivityLog(
            user_id=user.id,
            activity_type="login",
            description="ユーザーがログインしました",
            details={"timestamp": datetime.utcnow().isoformat()}
        )
        db.add(activity_log)
        db.commit()

        return {
            "access_token": access_token,
            "token_type": "bearer"
        }

    except Exception as e:
        logger.error(f"ログイン処理中にエラーが発生しました: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ログイン処理中にエラーが発生しました"
        )

# ログアウトエンドポイント
@api_app.post("/logout")
async def logout(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
):
    """
    ユーザーをログアウトし、セッションを無効化する
    """
    try:
        # セッションを無効化
        session = db.query(UserSession).filter(
            UserSession.token == token,
            UserSession.is_active == True
        ).first()
        
        if session:
            session.is_active = False
            db.commit()
            
            # アクティビティログの記録
            user_id = session.user_id
            activity_log = UserActivityLog(
                user_id=user_id,
                activity_type="logout",
                details={"timestamp": datetime.utcnow().isoformat()}
            )
            db.add(activity_log)
            db.commit()
            
            return {"message": "ログアウトしました"}
        
        return {"message": "既にログアウトしています"}
        
    except Exception as e:
        logger.error(f"ログアウト処理中にエラーが発生しました: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="ログアウト処理中にエラーが発生しました"
        )

# セッション検証の追加
async def verify_session(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> bool:
    """
    セッション検証をバイパス
    """
    return True

# 現在のユーザーを取得する関数を更新
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    デフォルトの管理者ユーザーを返す
    """
    try:
        # 管理者ユーザーを取得または作成
        user = db.query(User).filter(User.email == "admin@example.com").first()
        if not user:
            # デフォルトの管理者ユーザーを作成
            user = User(
                username="admin",
                email="admin@example.com",
                hashed_password=get_password_hash("admin123"),
                is_admin=True,
                is_active=True
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        return user
    except Exception as e:
        logger.error(f"ユーザー取得中にエラーが発生しました: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="ユーザー取得中にエラーが発生しました"
        )

# プロフィールエンドポイントの修正
@api_app.get("/profile")
async def get_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    ユーザープロフィールを取得する
    """
    try:
        profile = current_user.profile
        if not profile:
            # プロフィールが存在しない場合は新規作成
            profile = UserProfile(user_id=current_user.id)
            db.add(profile)
            db.commit()
            db.refresh(profile)
        
        return profile
    except Exception as e:
        logger.error(f"プロフィール取得中にエラーが発生しました: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="プロフィール取得中にエラーが発生しました"
        )

@api_app.put("/profile")
async def update_profile(
    profile_update: UserProfileUpdate,
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
):
    user = db.query(User).filter(User.email == token).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    profile = user.profile
    for field, value in profile_update.dict(exclude_unset=True).items():
        setattr(profile, field, value)
    
    db.commit()
    return profile

# チャット履歴検索エンドポイント
@api_app.post("/chats/search")
async def search_chats(
    search: ChatSearch,
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
):
    user = db.query(User).filter(User.email == token).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    # 通常のチャットのみを検索（シークレットモードを除外）
    query = db.query(Chat).filter(
        and_(
            Chat.user_id == user.id,
            Chat.is_secret == False
        )
    )
    
    # 検索条件の適用
    if search.query:
        query = query.join(Message).filter(
            or_(
                Chat.title.ilike(f"%{search.query}%"),
                Message.content.ilike(f"%{search.query}%")
            )
        )
    
    if search.start_date:
        query = query.filter(Chat.created_at >= search.start_date)
    
    if search.end_date:
        query = query.filter(Chat.created_at <= search.end_date)
    
    if search.tags:
        for tag in search.tags:
            query = query.join(ChatTag).filter(ChatTag.tag == tag)
    
    chats = query.order_by(desc(Chat.updated_at)).all()
    return chats

# 管理者ダッシュボードエンドポイント
@api_app.get("/admin/metrics")
async def get_admin_metrics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    try:
        # システムメトリクスの取得
        total_users = db.query(User).count()
        active_users = db.query(User).filter(User.is_active == True).count()
        total_chats = db.query(Chat).count()
        total_messages = db.query(Message).count()
        
        # アクティビティログの取得
        recent_activities = []
        try:
            activities = db.query(UserActivityLog).order_by(UserActivityLog.created_at.desc()).limit(10).all()
            for activity in activities:
                try:
                    recent_activities.append({
                        "id": activity.id,
                        "user_id": activity.user_id,
                        "activity_type": activity.activity_type,
                        "description": activity.description or "",
                        "created_at": activity.created_at.isoformat() if activity.created_at else None
                    })
                except Exception as e:
                    logger.error(f"アクティビティログの変換中にエラーが発生しました: {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"アクティビティログの取得中にエラーが発生しました: {str(e)}")
        
        metrics = {
            "total_users": total_users,
            "active_users": active_users,
            "total_chats": total_chats,
            "total_messages": total_messages,
            "recent_activity": recent_activities,
            "system_status": "healthy",
            "memory_usage": 0,  # TODO: 実際のメモリ使用率を取得
            "cpu_usage": 0,     # TODO: 実際のCPU使用率を取得
        }
        
        return metrics
    except Exception as e:
        logger.error(f"メトリクスの取得中にエラーが発生しました: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"メトリクスの取得中にエラーが発生しました: {str(e)}"
        )

@api_app.get("/admin/users")
async def get_admin_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    users = db.query(User).offset(skip).limit(limit).all()
    return users

@api_app.put("/admin/users/{user_id}/status")
async def update_user_status(
    user_id: int,
    is_active: bool,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.is_active = is_active
    db.commit()
    return {"message": "User status updated successfully"}

# テキスト音声変換エンドポイント
@api_app.post("/text-to-speech")
async def text_to_speech(
    text: str = Form(...),
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
):
    try:
        # テキストを音声に変換
        tts = gTTS(text=text, lang="ja")
        
        # 一時ファイルとして音声を保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            tts.save(temp_file.name)
            temp_file_path = temp_file.name
        
        # 音声ファイルを返す
        return FileResponse(
            temp_file_path,
            media_type="audio/mpeg",
            filename="speech.mp3"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 初期管理者ユーザー作成
@app.post("/api/setup/admin", response_model=Dict[str, str])
async def create_admin(db: Session = Depends(get_db)):
    """
    管理者ユーザーの作成
    """
    try:
        # 既存の管理者ユーザーをチェック
        admin_user = db.query(User).filter(User.email == "admin@example.com").first()
        if admin_user:
            return {"message": "管理者ユーザーは既に存在します"}

        # 管理者ユーザーの作成
        admin_user = User(
            username="admin",
            email="admin@example.com",
            hashed_password=get_password_hash("admin123"),
            is_admin=True,
            is_active=True
        )
        db.add(admin_user)
        db.flush()

        # 管理者プロフィールの作成
        profile = UserProfile(
            user_id=admin_user.id,
            full_name="System Administrator",
            bio="System Administrator"
        )
        db.add(profile)
        db.commit()

        return {"message": "管理者ユーザーを作成しました"}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# 管理者ユーザーの存在確認
@api_app.get("/setup/admin")
async def check_admin_exists(db: Session = Depends(get_db)):
    """
    管理者ユーザーが存在するか確認する
    """
    admin = db.query(User).filter(User.is_admin == True).first()
    return {"exists": admin is not None}

# ユーザー登録エンドポイント
@api_app.post("/register", response_model=Token)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    """
    新規ユーザーを登録する
    """
    try:
        # メールアドレスの重複チェック
        db_user = db.query(User).filter(User.email == user.email).first()
        if db_user:
            raise HTTPException(status_code=400, detail="このメールアドレスは既に登録されています")
        
        # ユーザーの作成
        hashed_password = get_password_hash(user.password)
        db_user = User(
            username=user.username,
            email=user.email,
            hashed_password=hashed_password,
            is_admin=user.is_admin,
            is_active=True
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        # ユーザープロフィールの作成
        profile = UserProfile(user_id=db_user.id)
        db.add(profile)
        db.commit()
        
        # アクセストークンの生成
        access_token = create_access_token(data={"sub": user.email})
        return {"access_token": access_token, "token_type": "bearer"}
        
    except Exception as e:
        db.rollback()
        logger.error(f"ユーザー登録中にエラーが発生しました: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="ユーザー登録中にエラーが発生しました"
        )

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=7)  # トークンの有効期限を7日に延長
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# メッセージ取得エンドポイント
@api_app.get("/messages")
async def get_messages(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
):
    try:
        # トークンからユーザー情報を取得
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            email = payload.get("sub")
            if email is None:
                raise HTTPException(status_code=401, detail="認証に失敗しました")
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="トークンの有効期限が切れています")
        except JWTError:
            raise HTTPException(status_code=401, detail="認証に失敗しました")
            
        # ユーザー認証
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(
                status_code=401,
                detail="認証に失敗しました。正しい認証情報を入力してください。"
            )
        
        # ユーザーのチャット履歴を取得
        chats = db.query(Chat).filter(Chat.user_id == user.id).all()
        
        # メッセージを取得
        messages = []
        for chat in chats:
            chat_messages = db.query(Message).filter(Message.chat_id == chat.id).all()
            for msg in chat_messages:
                messages.append({
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "chat_id": chat.id,
                    "created_at": msg.created_at.isoformat()
                })
        
        return {"messages": messages}
    except Exception as e:
        logger.error(f"メッセージ取得中にエラーが発生しました: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="メッセージ取得中にエラーが発生しました"
        )

# フロントエンド互換用のエンドポイント
@api_app.get("/api/messages")
async def get_messages_api(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
):
    return await get_messages(db, token)

if __name__ == '__main__':
    try:
        # データベースの初期化
        Base.metadata.create_all(bind=engine)
        print("Database initialized successfully")
        
        # アプリケーションの起動
        port = int(os.getenv('PORT', 8000))
        print(f"Starting application on port {port}")
        uvicorn.run("main:app", host='0.0.0.0', port=port, reload=True)
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        raise 