"""
ユーティリティ関数群
- パフォーマンス計測
- メモリ使用量のプロファイリング
- データベース接続の監視
- キャッシュの管理
"""

from functools import wraps
import time
from memory_profiler import profile
from sqlalchemy import event
from sqlalchemy.orm import Query
from monitoring import logger, CHAT_LATENCY, NLP_PROCESSING_TIME, DB_CONNECTIONS

def track_time(metric):
    """
    関数の実行時間を計測し、Prometheusメトリクスに記録するデコレータ
    
    Args:
        metric: 記録対象のPrometheusメトリクス
        
    Returns:
        function: デコレータ関数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            metric.observe(duration)
            return result
        return wrapper
    return decorator

@profile
def memory_profile(func):
    """
    関数のメモリ使用量をプロファイリングするデコレータ
    
    Args:
        func: プロファイリング対象の関数
        
    Returns:
        function: デコレータ関数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def setup_query_logging(engine):
    """
    SQLAlchemyのクエリログを設定
    - クエリの実行前にログを記録
    - クエリのパラメータも含めて記録
    
    Args:
        engine: SQLAlchemyエンジンインスタンス
    """
    @event.listens_for(Query, 'before_compile', retval=True)
    def log_query(query):
        logger.info('sql_query', 
                   query=str(query),
                   params=query.statement.compile(compile_kwargs={"literal_binds": True}))
        return query

def track_db_connections(engine):
    """
    データベース接続の監視
    - 接続の開始と終了を追跡
    - 接続数のカウント
    - 接続IDのログ記録
    
    Args:
        engine: SQLAlchemyエンジンインスタンス
    """
    @event.listens_for(engine, 'connect')
    def connect(dbapi_connection, connection_record):
        DB_CONNECTIONS.inc()
        logger.info('db_connection_opened', 
                   connection_id=id(dbapi_connection))

    @event.listens_for(engine, 'close')
    def close(dbapi_connection, connection_record):
        DB_CONNECTIONS.dec()
        logger.info('db_connection_closed', 
                   connection_id=id(dbapi_connection))

def cache_with_metrics(cache, timeout=300):
    """
    キャッシュのヒットとミスを追跡するデコレータ
    - キャッシュキーの生成
    - キャッシュの取得と設定
    - ログ記録
    
    Args:
        cache: キャッシュインスタンス
        timeout: キャッシュの有効期限（秒）
        
    Returns:
        function: デコレータ関数
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            cache_key = f.__name__ + str(args) + str(kwargs)
            rv = cache.get(cache_key)
            if rv is not None:
                logger.info('cache_hit', key=cache_key)
                return rv
            logger.info('cache_miss', key=cache_key)
            rv = f(*args, **kwargs)
            cache.set(cache_key, rv, timeout=timeout)
            return rv
        return decorated_function
    return decorator 