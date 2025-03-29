"""
自然言語処理（NLP）プロセッサー
- テキストの意図分類
- 感情分析
- エンティティ抽出
- 応答生成
- 表記ゆれ処理
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import numpy as np
from typing import Tuple, Dict, List, Union
import json
import os
from difflib import SequenceMatcher
import re
from collections import defaultdict

class NLPProcessor:
    """
    自然言語処理を担当するクラス
    - 日本語BERTモデルによる意図分類
    - 日本語GPT-2モデルによる応答生成
    - 感情分析とエンティティ抽出
    - 表記ゆれの処理と曖昧さの検出
    """
    
    def __init__(self):
        """
        初期化処理
        - 各種NLPモデルのロード
        - 意図とテンプレートの読み込み
        - 表記ゆれ辞書の初期化
        """
        # 日本語BERTモデルの初期化
        self.tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
        self.model = AutoModelForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese")
        
        # 日本語GPT-2モデルの初期化
        self.gpt_tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-medium")
        self.gpt_model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
        
        # 感情分析モデルの初期化
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="cl-tohoku/bert-base-japanese-sentiment")
        
        # エンティティ抽出モデルの初期化
        self.ner_model = pipeline("ner", model="cl-tohoku/bert-base-japanese-ner")
        
        # 意図の定義と応答テンプレートの読み込み
        self.load_intents_and_templates()
        
        # 表記ゆれ辞書の初期化
        self.variation_dict = self.load_variation_dict()
        
        # 曖昧さの閾値
        self.ambiguity_threshold = 0.7
        
    def load_intents_and_templates(self):
        """
        意図と応答テンプレートをJSONファイルから読み込む
        ファイルが存在しない場合はデフォルト値を設定
        """
        try:
            with open('intents.json', 'r', encoding='utf-8') as f:
                self.intents = json.load(f)
            with open('templates.json', 'r', encoding='utf-8') as f:
                self.templates = json.load(f)
        except FileNotFoundError:
            # デフォルトの意図とテンプレート
            self.intents = {
                "greeting": ["こんにちは", "おはよう", "こんばんは"],
                "farewell": ["さようなら", "バイバイ", "またね"],
                "question": ["?", "？", "教えて"],
                "thanks": ["ありがとう", "感謝", "サンキュー"],
                "complaint": ["不満", "文句", "困った"],
                "request": ["お願い", "頼む", "手伝って"]
            }
            self.templates = {
                'greeting': 'こんにちは！お手伝いできることはありますか？',
                'farewell': 'さようなら！またお会いしましょう。',
                'thanks': 'どういたしまして！他に何かお手伝いできることはありますか？',
                'help': '以下のようなことができます：\n- 質問への回答\n- 情報の検索\n- タスクの実行\n- 会話のサポート\n\n具体的にどのようなお手伝いが必要ですか？',
                'unknown': '申し訳ありません。よく分かりませんでした。もう少し詳しく説明していただけますか？',
                'error': '申し訳ありません。エラーが発生しました。もう一度お試しください。',
                'clarification': '以下のどのようなことについてお尋ねでしょうか？\n{options}',
                'processing': '処理中です。しばらくお待ちください。',
                'success': '処理が完了しました。他に何かお手伝いできることはありますか？'
            }
    
    def load_variation_dict(self) -> Dict[str, List[str]]:
        """
        表記ゆれ辞書の読み込み
        ファイルが存在しない場合はデフォルト値を設定
        
        Returns:
            Dict[str, List[str]]: 標準形とその表記ゆれの辞書
        """
        try:
            with open('variations.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # デフォルトの表記ゆれ辞書
            return {
                "こんにちは": ["こんにちわ", "こんにちはー", "こんにちは！"],
                "おはよう": ["おはようございます", "おはよー", "おはよ"],
                "さようなら": ["さよなら", "さようならー", "さよーなら"],
                "ありがとう": ["ありがと", "ありがとうございます", "ありがとー"],
                "教えて": ["教えてください", "教えて下さい", "教えてー"],
                "お願い": ["お願いします", "お願い致します", "お願いー"]
            }
    
    def normalize_text(self, text: str) -> str:
        """
        テキストの正規化処理
        - 全角英数字を半角に変換
        - 全角スペースを半角に変換
        - 連続する空白を1つに
        - 前後の空白を削除
        
        Args:
            text: 正規化対象のテキスト
            
        Returns:
            str: 正規化されたテキスト
        """
        # 全角英数字を半角に変換
        text = re.sub(r'[Ａ-Ｚａ-ｚ０-９]', lambda x: chr(ord(x.group(0)) - 0xFEE0), text)
        # 全角スペースを半角に変換
        text = text.replace('　', ' ')
        # 連続する空白を1つに
        text = re.sub(r'\s+', ' ', text)
        # 前後の空白を削除
        text = text.strip()
        return text
    
    def find_variations(self, text: str) -> List[str]:
        """
        表記ゆれの検出
        - 完全一致による検索
        - 類似度による検索
        
        Args:
            text: 検索対象のテキスト
            
        Returns:
            List[str]: 検出された表記ゆれのリスト
        """
        normalized_text = self.normalize_text(text)
        variations = []
        
        # 完全一致の検索
        for standard, variants in self.variation_dict.items():
            if normalized_text == standard or normalized_text in variants:
                variations.append(standard)
                variations.extend(variants)
        
        # 類似度による検索
        for standard, variants in self.variation_dict.items():
            for variant in [standard] + variants:
                similarity = SequenceMatcher(None, normalized_text, variant).ratio()
                if similarity > 0.8:  # 80%以上の類似度
                    variations.append(standard)
                    variations.extend(variants)
                    break
        
        return list(set(variations))
    
    def handle_ambiguity(self, text: str, intent: str, confidence: float) -> Tuple[str, float]:
        """
        曖昧な質問の処理
        - 信頼度が閾値以下の場合、関連する意図を検索
        
        Args:
            text: 入力テキスト
            intent: 検出された意図
            confidence: 信頼度
            
        Returns:
            Tuple[str, float]: 処理後の意図と信頼度
        """
        if confidence < self.ambiguity_threshold:
            # 曖昧な質問の場合、関連する意図を検索
            related_intents = self.find_related_intents(text)
            if related_intents:
                # 最も関連性の高い意図を選択
                best_intent = max(related_intents.items(), key=lambda x: x[1])
                return best_intent[0], best_intent[1]
        
        return intent, confidence
    
    def find_related_intents(self, text: str) -> Dict[str, float]:
        """
        関連する意図の検索
        - Jaccard類似度による関連性の計算
        
        Args:
            text: 入力テキスト
            
        Returns:
            Dict[str, float]: 意図と関連性スコアの辞書
        """
        related = {}
        text_tokens = set(self.tokenizer.tokenize(text))
        
        for intent, keywords in self.intents.items():
            # キーワードとの類似度を計算
            keyword_tokens = set()
            for keyword in keywords:
                keyword_tokens.update(self.tokenizer.tokenize(keyword))
            
            # Jaccard類似度の計算
            intersection = len(text_tokens & keyword_tokens)
            union = len(text_tokens | keyword_tokens)
            if union > 0:
                similarity = intersection / union
                if similarity > 0.3:  # 30%以上の類似度
                    related[intent] = similarity
        
        return related
    
    def generate_clarifying_question(self, text: str, related_intents: List[str]) -> str:
        """
        曖昧な質問に対する明確化の質問を生成
        
        Args:
            text: 入力テキスト
            related_intents: 関連する意図のリスト
            
        Returns:
            str: 明確化の質問
        """
        options = '\n'.join([f"- {intent}" for intent in related_intents])
        return self.templates['clarification'].format(options=options)
    
    def classify_intent(self, text: str) -> Tuple[str, float]:
        """
        テキストの意図を分類
        - テキストの正規化
        - 表記ゆれの検出
        - BERTモデルによる分類
        - 曖昧さの処理
        
        Args:
            text: 入力テキスト
            
        Returns:
            Tuple[str, float]: 検出された意図と信頼度
        """
        # テキストの正規化
        normalized_text = self.normalize_text(text)
        
        # 表記ゆれの検出
        variations = self.find_variations(normalized_text)
        if variations:
            normalized_text = variations[0]  # 標準形を使用
        
        # 入力テキストのトークン化
        inputs = self.tokenizer(normalized_text, return_tensors="pt", padding=True, truncation=True)
        
        # 予測の実行
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # 最も確率の高い意図を選択
        intent_idx = torch.argmax(predictions).item()
        confidence = predictions[0][intent_idx].item()
        
        # 意図のラベルを取得
        intent = list(self.intents.keys())[intent_idx]
        
        # 曖昧さの処理
        return self.handle_ambiguity(normalized_text, intent, confidence)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """
        テキストの感情分析を実行
        
        Args:
            text: 分析対象のテキスト
            
        Returns:
            Dict[str, Union[str, float]]: 感情分析の結果
        """
        sentiment = self.sentiment_analyzer(text)[0]
        return {
            'label': 'POSITIVE' if sentiment['label'] == 'POSITIVE' else 'NEGATIVE',
            'score': sentiment['score']
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        テキストからエンティティを抽出
        
        Args:
            text: 入力テキスト
            
        Returns:
            List[Dict[str, str]]: 抽出されたエンティティのリスト
        """
        return self.ner_model(text)
    
    def generate_response(self, text: str, intent: str, confidence: float) -> str:
        """
        応答の生成
        - 感情分析の実行
        - エンティティの抽出
        - 曖昧さの処理
        - 感情に基づく応答の調整
        - テンプレートベースの応答生成
        
        Args:
            text: 入力テキスト
            intent: 検出された意図
            confidence: 信頼度
            
        Returns:
            str: 生成された応答
        """
        # 感情分析の実行
        sentiment = self.analyze_sentiment(text)
        
        # エンティティの抽出
        entities = self.extract_entities(text)
        
        # 曖昧な質問の処理
        if confidence < self.ambiguity_threshold:
            related_intents = self.find_related_intents(text)
            if related_intents:
                return self.generate_clarifying_question(text, list(related_intents.keys()))
        
        # 感情に基づく応答の調整
        if sentiment['label'] == 'NEGATIVE' and sentiment['score'] > 0.8:
            return "申し訳ありません。より良いサービスを提供できるよう努めます。"
        
        # 意図に基づく応答の生成
        if intent in self.templates:
            response = self.templates[intent]
            
            # エンティティの置換
            for entity in entities:
                if entity['word'] in text:
                    response = response.replace('{entity}', entity['word'])
            
            return response
        
        # デフォルトの応答
        return "申し訳ありません。よく分かりません。もう少し具体的に教えていただけますか？"
    
    def save_intents_and_templates(self):
        """
        意図と応答テンプレートをJSONファイルに保存
        - intents.json: 意図の定義
        - templates.json: 応答テンプレート
        - variations.json: 表記ゆれ辞書
        """
        with open('intents.json', 'w', encoding='utf-8') as f:
            json.dump(self.intents, f, ensure_ascii=False, indent=2)
        with open('templates.json', 'w', encoding='utf-8') as f:
            json.dump(self.templates, f, ensure_ascii=False, indent=2)
        with open('variations.json', 'w', encoding='utf-8') as f:
            json.dump(self.variation_dict, f, ensure_ascii=False, indent=2) 