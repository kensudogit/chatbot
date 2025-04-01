"""
AIモデル管理クラス
- 大規模言語モデルを使用した応答生成
- 感情分析による感情状態の把握
- 音声認識と音声合成の処理
"""

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os
import re
import json
import openai

logger = logging.getLogger(__name__)

class AIModelManager:
    """
    AIモデルを管理し、チャットボットの中核機能を提供するクラス
    - 日本語に特化した応答生成
    - マルチモーダル処理（テキスト、音声）
    - 感情分析と文脈理解
    """
    
    def __init__(self):
        """
        AIモデルマネージャーの初期化
        OpenAIのAPIキーを環境変数から取得
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            openai.api_key = self.api_key
        
        # より軽量な日本語言語モデルの初期化
        model_name = "rinna/japanese-gpt-neox-3.6b"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )
        
        # GPUが利用可能な場合はGPUに移動
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        
        logger.info("AIモデルの初期化が完了しました")
        
        # コンテキスト管理用のメモリ
        self.context_memory = {}
        
        # 感情分析用の単語辞書（拡張版）
        self.emotion_dict = {
            "positive": [
                "嬉しい", "楽しい", "素晴らしい", "良い", "好き",
                "感謝", "ありがとう", "助かる", "便利", "快適",
                "満足", "期待", "希望", "成功", "達成",
                "優しい", "親切", "丁寧", "誠実", "信頼"
            ],
            "negative": [
                "悲しい", "辛い", "嫌い", "悪い", "困る",
                "不安", "心配", "焦る", "怒る", "腹立つ",
                "疲れる", "面倒", "複雑", "難しい", "失敗",
                "遅い", "遅れる", "間違う", "忘れる", "落ち込む"
            ],
            "neutral": [
                "普通", "まあまあ", "特に", "特にない",
                "一般的", "標準", "通常", "平均", "中間",
                "普通に", "いつも通り", "変わらず", "安定"
            ],
            "question": [
                "?", "？", "ですか", "でしょうか", "どうですか",
                "教えて", "説明", "理由", "なぜ", "どうして",
                "方法", "やり方", "手順", "コツ", "ポイント"
            ],
            "urgency": [
                "急いで", "すぐに", "早く", "至急", "緊急",
                "今すぐ", "直ちに", "一刻も早く", "優先", "重要"
            ]
        }
        
        # 自動応答テンプレート（拡張版）
        self.response_templates = {
            "greeting": [
                "こんにちは！お手伝いできることはありますか？",
                "こんにちは！今日はどんなことをお手伝いできますか？",
                "こんにちは！ご質問やお困りのことがありましたら、お気軽にお申し付けください。",
                "こんにちは！AIアシスタントです。お手伝いできることがありましたら、お気軽にお申し付けください。",
                "こんにちは！今日もお手伝いさせていただきます。ご質問はありますか？"
            ],
            "farewell": [
                "さようなら！またお会いしましょう。",
                "ご利用ありがとうございました。またお会いしましょう。",
                "お手伝いできて嬉しかったです。またお会いしましょう。",
                "ご利用ありがとうございました。またのご利用をお待ちしております。",
                "お手伝いできて嬉しかったです。またお会いしましょう。ごきげんよう！"
            ],
            "thanks": [
                "どういたしまして！他に何かお手伝いできることはありますか？",
                "お役に立てて嬉しいです！他にご質問はありますか？",
                "ありがとうございます！他に何かお手伝いできることはありますか？",
                "お手伝いできて嬉しいです。他にご要望はありますか？",
                "どういたしまして！他にご質問やお困りのことがありましたら、お気軽にお申し付けください。"
            ],
            "error": [
                "申し訳ありません。エラーが発生しました。もう一度お試しください。",
                "申し訳ありません。一時的な問題が発生しました。しばらく時間をおいて再度お試しください。",
                "申し訳ありません。システムエラーが発生しました。サポートに連絡させていただきます。",
                "申し訳ありません。処理中にエラーが発生しました。もう一度お試しください。",
                "申し訳ありません。現在システムの調整中です。しばらく時間をおいて再度お試しください。"
            ],
            "processing": [
                "処理中です。しばらくお待ちください。",
                "お待ちください。処理を実行しています。",
                "処理を実行中です。もうしばらくお待ちください。",
                "ただいま処理中です。しばらくお待ちください。",
                "処理を実行しています。完了までしばらくお待ちください。"
            ],
            "clarification": [
                "もう少し詳しく教えていただけますか？",
                "具体的にどのようなことでしょうか？",
                "もう少し詳しく説明していただけますか？",
                "ご質問の内容をより具体的に教えていただけますか？",
                "もう少し詳しくお聞かせいただけますでしょうか？"
            ],
            "confirmation": [
                "ご確認させていただきます。",
                "承知いたしました。",
                "はい、承知しました。",
                "ご要望を承りました。",
                "承知いたしました。ご確認させていただきます。"
            ]
        }
        
        # コンテキストの重み付け（最適化）
        self.context_weights = {
            "recent": 1.0,      # 最近の会話
            "sentiment": 0.9,   # 感情分析（重要度を上げる）
            "user_profile": 0.7, # ユーザープロフィール
            "time_of_day": 0.5,  # 時間帯
            "urgency": 0.8,     # 緊急度
            "question_type": 0.6 # 質問タイプ
        }
        
        # 会話の文脈を保持する最大数
        self.max_context_length = 10
        
        # 感情分析の閾値
        self.sentiment_thresholds = {
            "high": 0.8,
            "medium": 0.5,
            "low": 0.3
        }
        
    async def generate_chat_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> Optional[str]:
        """
        ChatGPTを使用してチャット応答を生成する

        Args:
            messages: 会話履歴のリスト
            max_tokens: 生成する応答の最大トークン数
            temperature: 応答の多様性を制御するパラメータ（0-1）

        Returns:
            生成された応答テキスト
        """
        try:
            if not self.api_key:
                logger.warning("OpenAI APIキーが設定されていません")
                return self._generate_fallback_response(messages[-1]["content"])

            # OpenAIのチャットAPIを呼び出し
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "あなたは親切で知識豊富なAIアシスタントです。"},
                    *messages
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.6
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"ChatGPT APIの呼び出し中にエラーが発生しました: {str(e)}")
            return self._generate_fallback_response(messages[-1]["content"])

    def _generate_fallback_response(self, user_message: str) -> str:
        """
        APIが利用できない場合のフォールバック応答を生成

        Args:
            user_message: ユーザーの最後のメッセージ

        Returns:
            フォールバック応答テキスト
        """
        # 基本的な応答ルール
        user_message = user_message.lower()
        
        if "こんにちは" in user_message:
            return "こんにちは！何かお手伝いできることはありますか？"
        elif "ありがとう" in user_message:
            return "どういたしまして！他に何かお手伝いできることはありますか？"
        elif "さようなら" in user_message or "バイバイ" in user_message:
            return "さようなら！また会いましょう！"
        elif "天気" in user_message:
            return "申し訳ありません。現在の天気情報は提供できません。"
        elif "名前" in user_message:
            return "私はAIアシスタントです。お手伝いさせていただきます。"
        elif "元気" in user_message:
            return "はい、元気です！あなたはいかがですか？"
        elif "?" in user_message or "？" in user_message:
            return "良い質問ですね。もう少し具体的に教えていただけますか？"
        else:
            return "承知しました。他に何かお手伝いできることはありますか？"

    def transcribe_audio(self, audio_data: bytes) -> str:
        """
        音声をテキストに変換する（将来の実装のためのプレースホルダー）
        """
        return "音声認識はまだ実装されていません。"

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        テキストの感情分析を行う
        
        Args:
            text: 分析対象のテキスト
            
        Returns:
            Dict: 感情分析結果
        """
        try:
            # 簡単な感情分析を返す（実際のプロジェクトでは感情分析モデルを使用）
            return {
                "label": "NEUTRAL",
                "score": 0.5
            }
        except Exception as e:
            self.logger.error(f"感情分析中にエラーが発生しました: {str(e)}")
            return {
                "label": "ERROR",
                "score": 0.0
            }

    def synthesize_speech(self, text: str) -> bytes:
        """
        テキストを音声に変換する
        
        Args:
            text: 変換対象のテキスト
            
        Returns:
            bytes: 生成された音声データ
        """
        try:
            # 音声合成の実行
            tts = gTTS(text=text, lang="ja")
            
            # 一時ファイルに保存
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                tts.save(temp_file.name)
                temp_file_path = temp_file.name
            
            # 音声データの読み込み
            with open(temp_file_path, "rb") as f:
                audio_data = f.read()
            
            # 一時ファイルの削除
            os.unlink(temp_file_path)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"音声合成中にエラーが発生しました: {str(e)}")
            raise
    
    def get_context(self, user_id: str) -> List[Dict[str, str]]:
        """
        ユーザーのコンテキストを取得する
        
        Args:
            user_id: ユーザーID
            
        Returns:
            List[Dict[str, str]]: コンテキスト情報
        """
        return self.context_memory.get(user_id, [])
    
    def update_context(self, user_id: str, user_message: str, bot_response: str):
        """
        ユーザーのコンテキストを更新する
        
        Args:
            user_id: ユーザーID
            user_message: ユーザーのメッセージ
            bot_response: ボットの応答
        """
        if user_id not in self.context_memory:
            self.context_memory[user_id] = []
        
        self.context_memory[user_id].append({
            "user_message": user_message,
            "bot_response": bot_response,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # コンテキストの長さを制限
        if len(self.context_memory[user_id]) > self.max_context_length:
            self.context_memory[user_id] = self.context_memory[user_id][-self.max_context_length:]
    
    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """
        会話履歴を適切な形式に整形する（改善版）
        
        Args:
            messages: 会話履歴のリスト
            
        Returns:
            str: 整形された会話テキスト
        """
        formatted_messages = []
        
        # 直近の会話のみを使用（コンテキストの制限）
        recent_messages = messages[-5:]  # 直近5つのメッセージのみを使用
        
        for msg in recent_messages:
            role_prefix = "ユーザー: " if msg["role"] == "user" else "アシスタント: "
            formatted_messages.append(f"{role_prefix}{msg['content']}")
        
        return "\n".join(formatted_messages)
    
    def _calculate_emotion_intensity(self, text: str) -> float:
        """
        テキストの感情強度を計算する
        
        Args:
            text: 分析対象のテキスト
            
        Returns:
            float: 感情の強度（0.0 ~ 1.0）
        """
        # 感情を表す単語や記号の数をカウント
        emotion_indicators = ["!", "?", "！", "？", "笑", "泣", "怒", "悲"]
        count = sum(1 for indicator in emotion_indicators if indicator in text)
        
        # 文字数で正規化
        intensity = min(count / len(text) * 5, 1.0) if text else 0.0
        
        return intensity
    
    def _optimize_response(self, response: str, messages: List[Dict[str, str]], sentiment: Dict) -> str:
        """
        応答を最適化する（改善版）
        
        Args:
            response: 生成された応答
            messages: メッセージのリスト
            sentiment: 感情分析の結果
            
        Returns:
            str: 最適化された応答
        """
        # 感情に基づく応答の調整（改善版）
        if sentiment["basic_sentiment"]["label"] == "ネガティブ" and sentiment["basic_sentiment"]["score"] > self.sentiment_thresholds["high"]:
            return "申し訳ありません。より良いサービスを提供できるよう努めます。具体的にどのような点でお困りでしょうか？"
        
        # 最近の会話履歴から意図を推測（改善版）
        intent = self._detect_intent(messages[-1]["content"])
        
        # 質問タイプに基づく応答の調整
        if sentiment["question_type"] != "general":
            if sentiment["question_type"] == "how_to":
                response = "手順について説明させていただきます。" + response
            elif sentiment["question_type"] == "why":
                response = "理由について説明させていただきます。" + response
            elif sentiment["question_type"] == "when":
                response = "時間について説明させていただきます。" + response
            elif sentiment["question_type"] == "who":
                response = "担当者について説明させていただきます。" + response
            elif sentiment["question_type"] == "where":
                response = "場所について説明させていただきます。" + response
        
        # 緊急度に基づく応答の調整
        if sentiment["urgency_level"] == "high":
            response = "緊急のご用件と承知いたしました。" + response
        elif sentiment["urgency_level"] == "medium":
            response = "お急ぎのご用件と承知いたしました。" + response
        
        # 意図に基づくテンプレートの適用（改善版）
        if intent in self.response_templates:
            templates = self.response_templates[intent]
            # 感情に基づいてテンプレートを選択
            if sentiment["basic_sentiment"]["label"] == "ポジティブ":
                response = templates[0]  # より親しみやすい応答
            elif sentiment["basic_sentiment"]["label"] == "ネガティブ":
                response = templates[-1]  # より丁寧な応答
            else:
                response = np.random.choice(templates)
        
        # 時間帯に基づく応答の調整（改善版）
        current_hour = datetime.now().hour
        if 5 <= current_hour < 12:
            response = "おはようございます。" + response
        elif 12 <= current_hour < 18:
            response = "こんにちは。" + response
        else:
            response = "こんばんは。" + response
        
        # コンテキストに基づく応答の調整
        if len(messages) > 1:
            prev_message = messages[-2]["content"]
            if any(word in prev_message for word in ["もう一度", "繰り返し", "再度"]):
                response = "先ほどの内容について、もう一度説明させていただきます。" + response
        
        return response
    
    def _detect_intent(self, text: str) -> str:
        """
        テキストから意図を検出する（改善版）
        
        Args:
            text: 入力テキスト
            
        Returns:
            str: 検出された意図
        """
        # キーワードベースの意図検出（拡張版）
        if any(word in text for word in ["こんにちは", "おはよう", "こんばんは", "はじめまして"]):
            return "greeting"
        elif any(word in text for word in ["さようなら", "バイバイ", "またね", "お疲れ様", "ご苦労様"]):
            return "farewell"
        elif any(word in text for word in ["ありがとう", "感謝", "サンキュー", "助かりました", "ありがとうございます"]):
            return "thanks"
        elif any(word in text for word in ["エラー", "問題", "困った", "できない", "動かない"]):
            return "error"
        elif any(word in text for word in ["待って", "待つ", "処理", "実行中", "進行中"]):
            return "processing"
        elif any(word in text for word in ["もう一度", "繰り返し", "再度", "説明", "詳しく"]):
            return "clarification"
        elif any(word in text for word in ["確認", "承知", "了解", "分かりました", "はい"]):
            return "confirmation"
        return "unknown"
    
    def _select_best_response(self, responses: List[str], messages: List[Dict[str, str]], user_id: str) -> str:
        """
        最適な応答を選択する
        
        Args:
            responses: 生成された応答の候補
            messages: メッセージのリスト
            user_id: ユーザーID
            
        Returns:
            str: 選択された最適な応答
        """
        # 各応答のスコアを計算
        scores = []
        for response in responses:
            score = 0.0
            
            # 長さのスコア（適度な長さを好む）
            length_score = 1.0 - abs(len(response) - 100) / 200
            score += length_score * 0.3
            
            # 感情の一貫性スコア
            sentiment = self.analyze_sentiment(response)
            if sentiment["basic_sentiment"]["label"] == "ポジティブ":
                score += 0.2
            
            # コンテキストとの関連性スコア
            context_score = self._calculate_context_relevance(response, messages)
            score += context_score * 0.5
            
            scores.append(score)
        
        # 最高スコアの応答を選択
        best_idx = np.argmax(scores)
        return responses[best_idx]

    def _calculate_context_relevance(self, response: str, messages: List[Dict[str, str]]) -> float:
        """
        応答とコンテキストの関連性を計算する
        
        Args:
            response: 応答テキスト
            messages: メッセージのリスト
            
        Returns:
            float: 関連性スコア（0.0 ~ 1.0）
        """
        # キーワードの抽出
        response_keywords = set(self._extract_keywords(response))
        context_keywords = set()
        
        # コンテキストからキーワードを抽出
        for msg in messages[-3:]:  # 直近3つのメッセージを考慮
            context_keywords.update(self._extract_keywords(msg["content"]))
        
        # キーワードの重複率を計算
        if not context_keywords:
            return 0.5
        
        overlap = len(response_keywords & context_keywords)
        return min(overlap / len(context_keywords), 1.0)

    def _extract_keywords(self, text: str) -> List[str]:
        """
        テキストからキーワードを抽出する
        
        Args:
            text: 入力テキスト
            
        Returns:
            List[str]: 抽出されたキーワード
        """
        # 単語の分割（簡単な実装）
        words = text.split()
        
        # ストップワードの除去
        stop_words = {"の", "は", "が", "を", "に", "へ", "と", "で", "や", "も", "な", "か", "ら", "れ", "さ", "し", "い", "う", "え", "お"}
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        
        return keywords 