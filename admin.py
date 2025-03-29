"""
管理画面インターフェース
- ユーザー管理
- チャット履歴管理
- 意図管理
- レスポンステンプレート管理
"""

from flask_admin.contrib.sqla import ModelView
from flask_login import current_user
from models import User, ChatHistory, Intent, ResponseTemplate
import json
from flask import redirect, url_for, request

class SecureModelView(ModelView):
    """
    セキュアなモデルビューの基本クラス
    - 管理者権限のチェック
    - アクセス制御
    """
    def is_accessible(self):
        """
        管理者権限のチェック
        
        Returns:
            bool: 管理者権限がある場合はTrue
        """
        if not current_user.is_authenticated:
            return False
        return current_user.is_admin

    def inaccessible_callback(self, name, **kwargs):
        """
        アクセス権限がない場合のコールバック
        
        Args:
            name: アクセスしようとしたビューの名前
            **kwargs: その他の引数
            
        Returns:
            Response: リダイレクトレスポンス
        """
        if not current_user.is_authenticated:
            return redirect(url_for('login', next=request.url))
        return redirect(url_for('index'))

    def _handle_view(self, name, **kwargs):
        """
        ビューアクセス時の処理
        
        Args:
            name: ビューの名前
            **kwargs: その他の引数
        """
        if not self.is_accessible():
            return self.inaccessible_callback(name, **kwargs)
        return super()._handle_view(name, **kwargs)

    def get_base_endpoint(self):
        """
        ベースエンドポイントの取得
        
        Returns:
            str: ベースエンドポイント名
        """
        return f"{self.endpoint_prefix}_{self.name.lower()}"

    def get_url(self, endpoint, **kwargs):
        """
        URLの生成
        
        Args:
            endpoint: エンドポイント名
            **kwargs: その他の引数
            
        Returns:
            str: 生成されたURL
        """
        if endpoint == 'index':
            return url_for('admin_index')
        return super().get_url(endpoint, **kwargs)

    def _get_view_url(self, name, **kwargs):
        """
        ビューのURLを取得
        
        Args:
            name: ビューの名前
            **kwargs: その他の引数
            
        Returns:
            str: 生成されたURL
        """
        if name == 'index':
            return url_for('admin_index')
        return super()._get_view_url(name, **kwargs)

    def _get_url_for_view(self, name, **kwargs):
        """
        ビューのURLを取得（オーバーライド）
        
        Args:
            name: ビューの名前
            **kwargs: その他の引数
            
        Returns:
            str: 生成されたURL
        """
        if name == 'index':
            return url_for('admin_index')
        return super()._get_url_for_view(name, **kwargs)

class UserAdmin(SecureModelView):
    """
    ユーザー管理画面
    - ユーザー情報の表示
    - 検索とフィルタリング
    - パスワードハッシュの非表示
    """
    column_list = ['id', 'username', 'email', 'created_at', 'is_admin']
    column_searchable_list = ['username', 'email']
    column_filters = ['created_at', 'is_admin']
    form_excluded_columns = ['password_hash', 'chat_history']
    can_create = True
    can_edit = True
    can_delete = True

class ChatHistoryAdmin(SecureModelView):
    """
    チャット履歴管理画面
    - 会話履歴の表示
    - 検索とフィルタリング
    - 読み取り専用
    """
    column_list = ['id', 'user', 'message', 'response', 'intent', 'confidence', 'timestamp']
    column_searchable_list = ['message', 'response']
    column_filters = ['intent', 'confidence', 'timestamp']
    can_create = False
    can_edit = False
    can_delete = False

class IntentAdmin(SecureModelView):
    """
    意図管理画面
    - 意図の定義管理
    - キーワードの表示
    - 検索とフィルタリング
    """
    column_list = ['id', 'name', 'description', 'is_active', 'created_at', 'updated_at']
    column_searchable_list = ['name', 'description']
    column_filters = ['is_active', 'created_at']
    can_create = True
    can_edit = True
    can_delete = True
    
    def _format_keywords(self, context, model, name):
        """
        キーワードの表示形式を整形
        
        Args:
            context: コンテキスト
            model: モデルインスタンス
            name: カラム名
            
        Returns:
            str: 整形されたキーワード文字列
        """
        if model.keywords:
            return ', '.join(json.loads(model.keywords))
        return ''
    
    column_formatters = {
        'keywords': _format_keywords
    }

class ResponseTemplateAdmin(SecureModelView):
    """
    レスポンステンプレート管理画面
    - テンプレートの管理
    - 変数の表示
    - 検索とフィルタリング
    """
    column_list = ['id', 'name', 'content', 'is_active', 'created_at', 'updated_at']
    column_searchable_list = ['name', 'content']
    column_filters = ['is_active', 'created_at']
    can_create = True
    can_edit = True
    can_delete = True
    
    def _format_variables(self, context, model, name):
        """
        変数の表示形式を整形
        
        Args:
            context: コンテキスト
            model: モデルインスタンス
            name: カラム名
            
        Returns:
            str: 整形された変数文字列
        """
        if model.variables:
            return ', '.join(json.loads(model.variables))
        return ''
    
    column_formatters = {
        'variables': _format_variables
    } 