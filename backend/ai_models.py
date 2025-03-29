"""
AIモデル管理クラス
- 大規模言語モデルを使用した応答生成
- 感情分析による感情状態の把握
- 音声認識と音声合成の処理
"""

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
import logging
import speech_recognition as sr
from gtts import gTTS
import tempfile
import os
import re

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
        初期化処理
        - 日本語言語モデルのロード
        - トークナイザーの設定
        - 感情分析モデルの初期化
        """
        try:
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
        except Exception as e:
            logger.error(f"AIモデルの初期化中にエラーが発生しました: {str(e)}")
            # エラーが発生しても、ダミーの応答を返せるようにする
            self.model = None
            self.tokenizer = None
        
        # コンテキスト管理用のメモリ
        self.context_memory = {}
        
        # 感情分析用の単語辞書
        self.emotion_dict = {
            "positive": ["嬉しい", "楽しい", "素晴らしい", "良い", "好き"],
            "negative": ["悲しい", "辛い", "嫌い", "悪い", "困る"],
            "neutral": ["普通", "まあまあ", "特に", "特にない"]
        }
        
    def generate_response(self, messages: List[Dict[str, str]], user_id: str) -> str:
        """
        メッセージに対する応答を生成する
        
        Args:
            messages: メッセージのリスト
            user_id: ユーザーID
            
        Returns:
            str: 生成された応答
        """
        try:
            if self.model is None or self.tokenizer is None:
                # モデルが利用できない場合は、ダミーの応答を返す
                return "申し訳ありません。現在AIモデルの初期化中です。しばらく時間をおいて再度お試しください。"
            
            # 会話履歴のフォーマット
            conversation = self._format_conversation(messages)
            
            # トークン化
            inputs = self.tokenizer(conversation, return_tensors="pt", padding=True)
            inputs = inputs.to(self.model.device)
            
            # 応答の生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 応答のデコード
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # コンテキストの更新
            self.update_context(user_id, messages[-1]["content"], response)
            
            return response
            
        except Exception as e:
            logger.error(f"応答生成中にエラーが発生しました: {str(e)}")
            return "申し訳ありません。応答の生成中にエラーが発生しました。"
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        テキストの感情分析を実行する
        
        Args:
            text: 分析対象のテキスト
            
        Returns:
            Dict[str, float]: 感情分析の結果（基本感情、強度、タイムスタンプ）
        """
        try:
            # 感情分析の実行
            pos_count = sum(1 for word in self.emotion_dict["positive"] if word in text)
            neg_count = sum(1 for word in self.emotion_dict["negative"] if word in text)
            
            # 感情スコアの計算
            total = pos_count + neg_count
            if total == 0:
                sentiment = {"label": "NEUTRAL", "score": 0.5}
            else:
                score = pos_count / (pos_count + neg_count)
                sentiment = {
                    "label": "POSITIVE" if score > 0.5 else "NEGATIVE",
                    "score": score if score > 0.5 else 1 - score
                }
            
            # 感情の強度を計算
            intensity = self._calculate_emotion_intensity(text)
            
            return {
                "basic_sentiment": {
                    "label": "ポジティブ" if sentiment["label"] == "POSITIVE" else "ネガティブ",
                    "score": sentiment["score"]
                },
                "intensity": intensity,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"感情分析中にエラーが発生しました: {str(e)}")
            raise
    
    def transcribe_audio(self, audio_data: bytes) -> str:
        """
        音声データをテキストに変換する
        
        Args:
            audio_data: 音声データのバイト列
            
        Returns:
            str: 変換されたテキスト
        """
        try:
            # 一時ファイルとして音声を保存
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # 音声認識の実行
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_file_path) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio, language="ja-JP")
            
            # 一時ファイルの削除
            os.unlink(temp_file_path)
            
            return text
            
        except Exception as e:
            logger.error(f"音声認識中にエラーが発生しました: {str(e)}")
            raise
    
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
        if len(self.context_memory[user_id]) > 10:
            self.context_memory[user_id] = self.context_memory[user_id][-10:]
    
    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """
        会話履歴を適切な形式に整形する
        
        Args:
            messages: 会話履歴のリスト
            
        Returns:
            str: 整形された会話テキスト
        """
        formatted_messages = []
        for msg in messages:
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