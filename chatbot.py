"""
チャットボットのメインアプリケーション
Flaskを使用したRESTful APIサーバー
- ユーザー認証
- チャット機能
- 会話履歴管理
- パフォーマンスモニタリング
"""

from flask import Flask, request, jsonify, redirect, url_for, render_template
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_caching import Cache
from flask_profiler import Profiler
# from flask_admin import Admin
from dotenv import load_dotenv
from models import db, User, ChatHistory, Intent, ResponseTemplate
from nlp import NLPProcessor
from celery import Celery
# from monitoring import setup_monitoring, logger, CHAT_REQUESTS, CHAT_LATENCY, NLP_PROCESSING_TIME, ACTIVE_USERS
from utils import track_time, memory_profile, setup_query_logging, track_db_connections, cache_with_metrics
# from admin import UserAdmin, ChatHistoryAdmin, IntentAdmin, ResponseTemplateAdmin
import os
from datetime import timedelta
import bcrypt
import asyncio
from functools import wraps
from sqlalchemy.orm import scoped_session, sessionmaker
from contextlib import contextmanager
import sentry_sdk
import logging

# ロガーの設定
logger = logging.getLogger(__name__)

# 環境変数の読み込み
load_dotenv()

# Flaskアプリケーションの初期化
app = Flask(__name__, 
    template_folder='templates',
    static_folder='static'
)

# Sentryの初期化（一時的に無効化）
print("Sentry initialization disabled for development")

# アプリケーション設定
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'mysql://chatbot_user:your_password@localhost/chatbot?charset=utf8mb4')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)

# キャッシュ設定（一時的に無効化）
app.config['CACHE_TYPE'] = 'simple'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300

# Celery設定（一時的に無効化）
app.config['CELERY_BROKER_URL'] = 'memory://'
app.config['CELERY_RESULT_BACKEND'] = 'memory://'

# プロファイラー設定
app.config['flask_profiler'] = {
    'enabled': True,
    'storage': {
        'engine': 'sqlite',
        'db_location': 'profiler.db'
    },
    'basicAuth': {
        'enabled': True,
        'username': 'admin',
        'password': 'admin'
    }
}

# 拡張機能の初期化
db.init_app(app)
with app.app_context():
    db.create_all()

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'ログインが必要です'
login_manager.login_message_category = 'info'
jwt = JWTManager(app)
cache = Cache(app)
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)
# profiler = Profiler(app)  # 一時的に無効化

# 管理画面の初期化（コメントアウト）
# admin = Admin(app, name='チャットボット管理画面', template_mode='bootstrap4', url='/admin', index_view=None, base_template='admin/base.html')

# モニタリングの初期化（一時的に無効化）
# metrics = setup_monitoring(app)

# NLPプロセッサーの初期化
nlp_processor = NLPProcessor()

# データベースセッション管理とモニタリングの設定
session_factory = None
Session = None

def init_db_session():
    global session_factory, Session
    session_factory = sessionmaker(bind=db.engine)
    Session = scoped_session(session_factory)
    # データベースモニタリングの設定（一時的に無効化）
    # setup_query_logging(db.engine)
    # track_db_connections(db.engine)

with app.app_context():
    init_db_session()

# 管理画面の登録（コメントアウト）
# admin.add_view(UserAdmin(User, db.session))
# admin.add_view(ChatHistoryAdmin(ChatHistory, db.session))
# admin.add_view(IntentAdmin(Intent, db.session))
# admin.add_view(ResponseTemplateAdmin(ResponseTemplate, db.session))

# 管理画面のルーティングを明示的に設定（コメントアウト）
# @app.route('/admin')
# @login_required
# def admin_index():
#     """
#     管理画面のインデックスページ
#     """
#     if not current_user.is_admin:
#         return redirect(url_for('index'))
#     return admin.index()

# インデックスページのルーティングを追加
@app.route('/')
def index():
    """
    インデックスページ
    """
    if current_user.is_authenticated:
        return redirect(url_for('chat'))
    return redirect(url_for('login'))

@contextmanager
def session_scope():
    """
    データベースセッションのコンテキストマネージャー
    トランザクションの自動コミットとロールバックを管理
    """
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def async_route(f):
    """
    非同期ルートハンドラーを同期関数に変換するデコレータ
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper

@login_manager.user_loader
@cache_with_metrics(cache, timeout=300)
def load_user(user_id):
    """
    ユーザー情報をキャッシュ付きで読み込む
    """
    return User.query.get(int(user_id))

@app.route('/register', methods=['POST'])
# @track_time(CHAT_LATENCY)
def register():
    """
    新規ユーザー登録エンドポイント
    - ユーザー名とメールアドレスの重複チェック
    - パスワードのハッシュ化
    """
    data = request.get_json()
    
    with session_scope() as session:
        if session.query(User).filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already exists'}), 400
            
        if session.query(User).filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already exists'}), 400
        
        password_hash = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())
        
        new_user = User(
            username=data['username'],
            email=data['email'],
            password_hash=password_hash.decode('utf-8')
        )
        
        session.add(new_user)
        logger.info('user_registered', extra={'username': data['username']})
        return jsonify({'message': 'User registered successfully'}), 201

@app.route('/login', methods=['GET', 'POST'])
# @track_time(CHAT_LATENCY)
def login():
    """
    ユーザーログインエンドポイント
    - 認証情報の検証
    - セッション管理
    - JWTトークンの生成
    """
    if request.method == 'GET':
        return render_template('login.html')
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400
    
    user = User.query.filter_by(username=username).first()
    
    if user and bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
        login_user(user)
        access_token = create_access_token(identity=user.id)
        # ACTIVE_USERS.inc()
        logger.info('user_logged_in', extra={'username': username})
        
        response_data = {
            'status': 'success',
            'message': 'Login successful',
            'access_token': access_token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin
            }
        }
        return jsonify(response_data), 200
    
    logger.warning('login_failed', extra={'username': username})
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/logout')
@login_required
def logout():
    """
    ログアウトエンドポイント
    - セッションのクリア
    - アクティブユーザー数の更新
    """
    logout_user()
    # ACTIVE_USERS.dec()
    logger.info('user_logged_out', username=current_user.username if current_user.is_authenticated else 'unknown')
    return jsonify({'message': 'Logged out successfully'}), 200

@celery.task
@memory_profile
def process_chat_message(user_id, message, intent, confidence, response):
    """
    チャットメッセージを非同期で処理するCeleryタスク
    - 会話履歴の保存
    - メモリ使用量のプロファイリング
    """
    with session_scope() as session:
        chat_history = ChatHistory(
            user_id=user_id,
            message=message,
            response=response,
            intent=intent,
            confidence=confidence
        )
        session.add(chat_history)
        logger.info('chat_message_processed', 
                   user_id=user_id,
                   intent=intent,
                   confidence=confidence)

@app.route('/chat', methods=['POST'])
@jwt_required()
@async_route
# @track_time(CHAT_LATENCY)
async def chat():
    """
    チャットメッセージ処理エンドポイント
    - JWT認証
    - 非同期処理
    - レイテンシ計測
    """
    # CHAT_REQUESTS.inc()
    user_id = get_jwt_identity()
    data = request.get_json()
    user_message = data.get('message', '').strip()
    
    # NLPによるメッセージ処理
    intent, confidence = await asyncio.to_thread(
        nlp_processor.classify_intent,
        user_message
    )
    response = await asyncio.to_thread(
        nlp_processor.generate_response,
        user_message,
        intent,
        confidence
    )
    
    # 非同期でチャット履歴を保存
    process_chat_message.delay(user_id, user_message, intent, confidence, response)
    
    logger.info('chat_response_generated',
                extra={
                    'user_id': user_id,
                    'intent': intent,
                    'confidence': confidence
                })
    
    return jsonify({
        'response': response,
        'intent': intent,
        'confidence': confidence,
        'status': 'success'
    })

@app.route('/chat/history', methods=['GET'])
@jwt_required()
# @cache_with_metrics(cache, timeout=60)
def get_chat_history():
    """
    チャット履歴取得エンドポイント
    - JWT認証
    - キャッシュ付き
    - 最新10件の履歴を返却
    """
    user_id = get_jwt_identity()
    with session_scope() as session:
        history = session.query(ChatHistory)\
            .filter_by(user_id=user_id)\
            .order_by(ChatHistory.timestamp.desc())\
            .limit(10)\
            .all()
        
        return jsonify({
            'history': [{
                'message': h.message,
                'response': h.response,
                'timestamp': h.timestamp.isoformat(),
                'intent': h.intent,
                'confidence': h.confidence
            } for h in history]
        })

@app.route('/health', methods=['GET'])
def health_check():
    """
    ヘルスチェックエンドポイント
    - データベース接続状態の確認
    """
    try:
        db.engine.connect()
        db_status = 'connected'
    except Exception as e:
        db_status = f'disconnected: {str(e)}'
        logger.error('database_connection_error', error=str(e))
    
    return jsonify({
        'status': 'healthy',
        'database': db_status
    })

@app.route('/create-admin', methods=['POST'])
def create_admin():
    """
    管理者アカウント作成エンドポイント
    - ユーザー名とメールアドレスの重複チェック
    - パスワードのハッシュ化
    - 管理者権限の付与
    """
    data = request.get_json()
    
    with session_scope() as session:
        if session.query(User).filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already exists'}), 400
            
        if session.query(User).filter_by(email=data['email']).first():
            return jsonify({'error': 'Email already exists'}), 400
        
        password_hash = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())
        
        new_admin = User(
            username=data['username'],
            email=data['email'],
            password_hash=password_hash.decode('utf-8'),
            is_admin=True
        )
        
        session.add(new_admin)
        logger.info('admin_user_created', username=data['username'])
        return jsonify({'message': 'Admin user created successfully'}), 201

if __name__ == '__main__':
    try:
        with app.app_context():
            # データベースの初期化
            db.create_all()
            print("Database initialized successfully")
            
            # 管理者アカウントの作成（存在しない場合）
            with session_scope() as session:
                admin = session.query(User).filter_by(username='admin').first()
                if not admin:
                    password_hash = bcrypt.hashpw('admin123'.encode('utf-8'), bcrypt.gensalt())
                    admin = User(
                        username='admin',
                        email='admin@example.com',
                        password_hash=password_hash.decode('utf-8'),
                        is_admin=True
                    )
                    session.add(admin)
                    print("Admin user created successfully")
                else:
                    print("Admin user already exists")
        
        # アプリケーションの起動
        port = int(os.getenv('PORT', 8000))
        print(f"Starting application on port {port}")
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        raise 