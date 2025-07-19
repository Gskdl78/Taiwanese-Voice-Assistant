#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
羅馬拼音格式轉換器
將多種格式的台語羅馬拼音（包含聲調符號）轉換為數字調格式。
支援聲母 o͘ 和輕聲 (--)。
"""

import re
import unicodedata

class RomanizationConverter:
    """羅馬拼音格式轉換器"""
    
    def __init__(self):
        """初始化聲調映射表"""
        # Unicode 組合聲調符號 -> 數字調
        self.tone_map = {
            '\u0301': '2',  # Combining Acute Accent (ˊ)
            '\u0300': '3',  # Combining Grave Accent (ˋ)
            '\u0302': '5',  # Combining Circumflex Accent (ˆ)
            '\u0304': '7',  # Combining Macron (ˉ)
            '\u030d': '8',  # Combining Vertical Line Above (̍)
        }
        # o͘ 的組合點符號
        self.o_dot_char = '\u0358'  # Combining Dot Above Right

    def convert_to_numeric_tone(self, romanization_text):
        """
        將羅馬拼音轉換為數字調格式
        
        Args:
            romanization_text (str): 輸入的羅馬拼音文字（如 "guá sī o͘-á"）
            
        Returns:
            str: 轉換後的數字調格式（如 "gua2 si7 oo-a2"）
        """
        import time
        
        start_time = time.time()
        try:
            print(f"⏱️  開始羅馬拼音轉換: '{romanization_text}'")
            
            clean_start = time.time()
            cleaned_text = self._clean_romanization(romanization_text)
            clean_time = time.time() - clean_start
            print(f"　├─ 文字清理耗時: {clean_time:.3f}秒，結果: '{cleaned_text}'")
            
            convert_start = time.time()
            converted_text = self._convert_syllables(cleaned_text)
            convert_time = time.time() - convert_start
            total_time = time.time() - start_time
            print(f"　├─ 聲調轉換耗時: {convert_time:.3f}秒")
            print(f"✅ 羅馬拼音轉換完成，總耗時: {total_time:.3f}秒，結果: '{converted_text}'")
            
            return converted_text
            
        except Exception as e:
            total_time = time.time() - start_time
            print(f"❌ 羅馬拼音轉換錯誤，總耗時: {total_time:.3f}秒，錯誤: {e}")
            return romanization_text
    
    def _clean_romanization(self, text):
        """清理羅馬拼音文字，移除不必要的標點符號"""
        text = re.sub(r'[，。！？；：,.?!;:]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _convert_syllables(self, text):
        """將文字分割成音節並逐個轉換"""
        parts = re.split(r'(\s+|-)', text)
        converted_parts = [self._convert_single_syllable(part) for part in parts]
        return ''.join(converted_parts)
    
    def _convert_single_syllable(self, syllable):
        """
        轉換單個音節的核心邏輯
        包含聲調、o͘ 和輕聲處理
        """
        if not syllable or re.match(r'^\s+$', syllable) or syllable == '-' or re.search(r'\d$', syllable):
            return syllable

        # 1. 處理輕聲 (neutral tone)
        is_neutral_tone = False
        if syllable.startswith('--'):
            is_neutral_tone = True
            syllable = syllable[2:]

        # 2. 使用 NFD 正規化，分解為基礎字符和組合符號
        decomposed = unicodedata.normalize('NFD', syllable)
        
        base_chars = []
        tone_char = None

        for char in decomposed:
            if unicodedata.combining(char):
                if char in self.tone_map and not tone_char:
                    tone_char = char
                else:
                    # 保留其他組合符號，例如 o͘ 的點
                    base_chars.append(char)
            else:
                base_chars.append(char)
        
        # 3. 重建基礎音節，並處理特殊母音 o͘
        # 將 'o' + 'combining dot' 替換為 'oo'
        base_syllable = "".join(base_chars).replace(f'o{self.o_dot_char}', 'oo')

        # 4. 決定聲調數字
        tone_number = ""
        if is_neutral_tone:
            tone_number = '0'
        elif tone_char:
            tone_number = self.tone_map[tone_char]
        else:
            # 預設聲調規則 (入聲字 vs. 一般字)
            if base_syllable.lower().endswith(('p', 't', 'k', 'h')):
                tone_number = '4'
            elif any(c in 'aeiou' for c in base_syllable.lower()):
                tone_number = '1'

        return base_syllable + tone_number
    
    def test_conversion(self):
        """擴充後的測試案例"""
        test_cases = [
            "guá sī Tâi-oân-lâng",   # 基本聲調
            "lí hó",                  # 基本聲調
            "tsia̍h-pn̄g",             # 第8聲和第7聲
            "kám-siā",               # 第2聲和第1聲
            "o͘-á-kah",               # 關鍵的 o͘ 處理
            "tsòe-kang",             # o͘e 的組合 (實際是 tsuè)
            "goe̍h-niû",              # e̍h 的組合
            "--lah",                 # 輕聲處理
            "tsit-ê --lâng chin-kán-tan", # 句子中的輕聲
            "thak-tsheh",            # 入聲字 (p,t,k,h結尾)
            "sió-bōo",               # 第3聲
            "sann",                  # 單音節
            "Tâi-lô"                 # 已經是數字調(此處應為羅馬字)
        ]
        
        print("\n=== 羅馬拼音轉換器升級版測試 ===")
        for test_text in test_cases:
            result = self.convert_to_numeric_tone(test_text)
            print(f"輸入: {test_text}")
            print(f"輸出: {result}")
            print("-" * 40)

# 測試用的主程式
if __name__ == "__main__":
    converter = RomanizationConverter()
    converter.test_conversion() 