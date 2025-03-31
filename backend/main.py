from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, ConfigDict
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import uvicorn
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, desc, create_engine
import speech_recognition as sr
from gtts import gTTS
import os
import tempfile
import shutil
import json
import logging
import jwt
from jose import JWTError

# データベース設定
SQLALCHEMY_DATABASE_URL = "sqlite:///chatbot.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=True
)

from database import (
    get_db, User, Chat, Message as DBMessage, UserProfile, UserActivityLog,
    SystemMetrics, ChatTag, verify_password, get_password_hash, create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES, SECRET_KEY, ALGORITHM, UserSession
)
from ai_models import AIModelManager

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切なオリジンに制限することを推奨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,  # プリフライトリクエストのキャッシュ時間（秒）
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
@api_app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme),
    user_agent: str = Header(None)
):
    try:
        # デバイスタイプの判定
        device_type = get_device_type(user_agent or "")
        
        # トークンからユーザー情報を取得
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            email = payload.get("sub")
            if email is None:
                raise HTTPException(status_code=401, detail="認証に失敗しました")
        except JWTError:
            raise HTTPException(status_code=401, detail="認証に失敗しました")
            
        # ユーザー認証
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(
                status_code=401,
                detail="認証に失敗しました。正しい認証情報を入力してください。"
            )
        
        # AIモデルによる応答生成
        response = ai_manager.generate_response(request.messages, user.id)
        
        # モバイルデバイスの場合、応答を最適化
        if device_type == "mobile":
            response = response[:500] + ("..." if len(response) > 500 else "")
        
        # 感情分析
        sentiment = ai_manager.analyze_sentiment(request.messages[-1].content)
        
        # チャット履歴の保存
        try:
            chat = Chat(
                user_id=user.id,
                title=request.title or "新しいチャット",
                is_secret=request.is_secret
            )
            db.add(chat)
            db.commit()
            
            if not request.is_secret:
                for msg in request.messages:
                    db_message = DBMessage(
                        chat_id=chat.id,
                        role=msg.role,
                        content=msg.content,
                        sentiment=str(sentiment)
                    )
                    db.add(db_message)
                db.commit()
                logger.info(f"チャット履歴を保存しました。ユーザーID: {user.id}, チャットID: {chat.id}")
            else:
                logger.info(f"シークレットモードでチャットが実行されました。ユーザーID: {user.id}")
        except Exception as e:
            logger.error(f"チャット履歴の保存中にエラーが発生しました: {str(e)}")
            db.rollback()
            # エラーが発生しても、応答は返す
            pass
        
        return ChatResponse(
            response=response,
            sentiment=sentiment,
            device_type=device_type
        )
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="サーバーエラーが発生しました。しばらく時間をおいて再度お試しください。"
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

# トークン取得エンドポイント
@api_app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    ユーザー認証とアクセストークンの発行
    """
    try:
        # ユーザーの検証
        user = db.query(User).filter(User.email == form_data.username).first()
        if not user or not verify_password(form_data.password, user.hashed_password):
            raise HTTPException(
                status_code=401,
                detail="メールアドレスまたはパスワードが正しくありません",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=401,
                detail="このアカウントは無効です",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # アクセストークンの生成
        access_token = create_access_token(
            data={"sub": user.email},
            expires_delta=ACCESS_TOKEN_EXPIRE_MINUTES
        )
        
        # セッションの作成
        expires_at = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        session = UserSession(
            user_id=user.id,
            token=access_token,
            expires_at=expires_at
        )
        db.add(session)
        
        # ユーザーのログイン時間を更新
        user.last_login = datetime.utcnow()
        db.commit()
        
        return {"access_token": access_token, "token_type": "bearer"}
        
    except Exception as e:
        logger.error(f"ログイン処理中にエラーが発生しました: {str(e)}")
        raise HTTPException(
            status_code=500,
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
    セッションの有効性を検証する
    """
    session = db.query(UserSession).filter(
        UserSession.token == token,
        UserSession.is_active == True,
        UserSession.expires_at > datetime.utcnow()
    ).first()
    
    if not session:
        raise HTTPException(
            status_code=401,
            detail="セッションが無効です。再度ログインしてください。",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # セッションの最終アクティビティを更新
    session.last_activity = datetime.utcnow()
    db.commit()
    
    return True

# 現在のユーザーを取得する関数を更新
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    トークンからユーザーを取得し、セッションを検証する
    """
    try:
        # セッションの検証
        await verify_session(token, db)
        
        # トークンの検証とユーザーの取得
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(
                status_code=401,
                detail="認証に失敗しました",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(
                status_code=401,
                detail="認証に失敗しました",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user
        
    except Exception as e:
        logger.error(f"ユーザー認証中にエラーが発生しました: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="認証に失敗しました",
            headers={"WWW-Authenticate": "Bearer"},
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
        query = query.join(DBMessage).filter(
            or_(
                Chat.title.ilike(f"%{search.query}%"),
                DBMessage.content.ilike(f"%{search.query}%")
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
        total_messages = db.query(DBMessage).count()
        
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
@api_app.post("/setup/admin")
async def create_admin(db: Session = Depends(get_db)):
    """
    初期管理者ユーザーを作成する
    """
    try:
        # 既存の管理者ユーザーをチェック
        admin = db.query(User).filter(User.is_admin == True).first()
        if admin:
            raise HTTPException(status_code=400, detail="管理者ユーザーは既に存在します")
        
        # 管理者ユーザーを作成
        hashed_password = get_password_hash("admin123")
        admin_user = User(
            username="admin",
            email="admin@example.com",
            hashed_password=hashed_password,
            is_admin=True,
            is_active=True
        )
        
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)
        
        # 管理者プロフィールを作成
        profile = UserProfile(user_id=admin_user.id)
        db.add(profile)
        db.commit()
        
        return {"message": "管理者ユーザーが正常に作成されました", "user_id": admin_user.id}
        
    except Exception as e:
        db.rollback()
        logger.error(f"管理者ユーザーの作成中にエラーが発生しました: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="管理者ユーザーの作成中にエラーが発生しました"
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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 