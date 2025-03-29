"""
モニタリングとロギング設定
- Prometheusメトリクス
- 構造化ログ
"""

from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge
import structlog
import os
from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# Prometheusメトリクスの初期化
metrics = PrometheusMetrics.for_app_factory()

# カスタムメトリクスの定義
CHAT_REQUESTS = Counter('chat_requests_total', 'チャットリクエストの総数')
CHAT_LATENCY = Histogram('chat_latency_seconds', 'チャットリクエストの応答時間（秒）')
NLP_PROCESSING_TIME = Histogram('nlp_processing_seconds', 'NLP処理時間（秒）')
ACTIVE_USERS = Gauge('active_users', 'アクティブユーザー数')
DB_CONNECTIONS = Gauge('db_connections', 'アクティブなデータベース接続数')
CACHE_HITS = Counter('cache_hits_total', 'キャッシュヒットの総数')
CACHE_MISSES = Counter('cache_misses_total', 'キャッシュミスの総数')

# 構造化ログの設定
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),  # ISO形式のタイムスタンプ
        structlog.processors.JSONRenderer()  # JSON形式での出力
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    wrapper_class=structlog.BoundLogger
)

# ロガーの初期化
logger = structlog.get_logger()

def setup_monitoring(app):
    """
    アプリケーションのモニタリングとロギングを初期化
    
    Args:
        app: Flaskアプリケーションインスタンス
        
    Returns:
        PrometheusMetrics: メトリクスインスタンス
    """
    metrics.init_app(app)
    
    # デフォルトメトリクスの追加
    metrics.info('app_info', 'アプリケーション情報', version='1.0.0')
    
    # カスタムメトリクスエンドポイントの追加
    @app.route('/metrics/custom')
    def custom_metrics():
        """
        カスタムメトリクスを取得するエンドポイント
        
        Returns:
            str: Prometheus形式のメトリクスデータ
        """
        return metrics.generate_latest()
    
    return metrics 