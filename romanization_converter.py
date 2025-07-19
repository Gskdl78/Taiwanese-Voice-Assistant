#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
羅馬拼音格式轉換器
將意傳科技API的羅馬拼音轉換為數字調格式
"""

import re
import unicodedata

class RomanizationConverter:
    """羅馬拼音格式轉換器"""
    
    def __init__(self):
        # 台語聲調符號對數字的映射
        self.tone_marks = {
            # 第1聲 (陰平) - 通常不標調
            # 第2聲 (陰上) - 銳音符 ́
            'á': '2', 'é': '2', 'í': '2', 'ó': '2', 'ú': '2', 'ń': '2', 'ḿ': '2',
            # 第3聲 (陰去) - 重音符 ̀
            'à': '3', 'è': '3', 'ì': '3', 'ò': '3', 'ù': '3', 'ǹ': '3', 'm̀': '3',
            # 第5聲 (陽平) - 抑揚符 ̂
            'â': '5', 'ê': '5', 'î': '5', 'ô': '5', 'û': '5', 'n̂': '5', 'm̂': '5',
            # 第7聲 (陽去) - 長音符 ̄
            'ā': '7', 'ē': '7', 'ī': '7', 'ō': '7', 'ū': '7', 'n̄': '7', 'm̄': '7',
            # 第8聲 (陽入) - 點下方 ̍
            'a̍': '8', 'e̍': '8', 'i̍': '8', 'o̍': '8', 'u̍': '8', 'n̍': '8', 'm̍': '8',
        }
        
    def convert_to_numeric_tone(self, romanization_text):
        """
        將羅馬拼音轉換為數字調格式
        
        Args:
            romanization_text (str): 輸入的羅馬拼音文字（如 "guá sī kò sió tsōo-tshiú"）
            
        Returns:
            str: 轉換後的數字調格式（如 "gua2 si7 ko3 sio2 tsoo7-tshiu2"）
        """
        import time
        
        start_time = time.time()
        try:
            print(f"⏱️  開始羅馬拼音轉換: '{romanization_text}'")
            
            # 清理輸入文字
            clean_start = time.time()
            cleaned_text = self._clean_romanization(romanization_text)
            clean_time = time.time() - clean_start
            print(f"　├─ 文字清理耗時: {clean_time:.3f}秒，結果: '{cleaned_text}'")
            
            # 轉換聲調符號
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
            # 如果轉換失敗，返回原文字
            return romanization_text
    
    def _clean_romanization(self, text):
        """清理羅馬拼音文字"""
        # 移除多餘的空格和標點符號
        text = re.sub(r'[，。！？；：]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _convert_syllables(self, text):
        """將文字分割成音節並逐個轉換"""
        # 分割成音節（以空格和連字符分割）
        parts = re.split(r'(\s+|-)', text)
        converted_parts = []
        
        for part in parts:
            if re.match(r'^\s+$', part) or part == '-':
                # 保留空格和連字符
                converted_parts.append(part)
            else:
                # 轉換音節
                converted_syllable = self._convert_single_syllable(part)
                converted_parts.append(converted_syllable)
        
        return ''.join(converted_parts)
    
    def _convert_single_syllable(self, syllable):
        """轉換單個音節"""
        if not syllable:
            return syllable
            
        # 檢查是否已經包含數字（已經是數字調格式）
        if re.search(r'\d', syllable):
            return syllable
            
        # 尋找聲調符號 - 需要檢查完整的組合字符
        tone_found = False
        result = syllable
        
        # 檢查是否包含聲調符號（使用字符串匹配而不是單個字符）
        for tone_char, tone_num in self.tone_marks.items():
            if tone_char in syllable:
                # 移除聲調符號，加上數字
                base_char = self._remove_tone_mark(tone_char)
                result = result.replace(tone_char, base_char)
                result = result + tone_num
                tone_found = True
                break
        
        # 如果沒有找到聲調符號，判斷是否需要加上預設聲調
        if not tone_found:
            # 檢查是否為入聲字（以 h, t, k, p 結尾）
            if syllable.endswith(('h', 't', 'k', 'p')):
                # 入聲字通常是第4聲或第8聲，這裡預設第4聲
                result = syllable + '4'
            elif syllable and any(c in syllable for c in 'aeiou'):
                # 有母音但無聲調符號，預設第1聲
                result = syllable + '1'
            else:
                # 其他情況保持原樣
                result = syllable
        
        return result
    
    def _remove_tone_mark(self, char):
        """移除聲調符號，返回基本字符"""
        # 使用 Unicode 正規化來移除聲調符號
        normalized = unicodedata.normalize('NFD', char)
        base_char = ''.join(c for c in normalized if not unicodedata.combining(c))
        return base_char
    
    def test_conversion(self):
        """測試轉換功能"""
        test_cases = [
            "guá sī kò sió tsōo-tshiú",
            "lí hó",
            "tsia̍h-pn̄g",
            "kám-siā"
        ]
        
        print("=== 羅馬拼音轉換測試 ===")
        for test_text in test_cases:
            result = self.convert_to_numeric_tone(test_text)
            print(f"輸入: {test_text}")
            print(f"輸出: {result}")
            print("-" * 40)

# 測試用的主程式
if __name__ == "__main__":
    converter = RomanizationConverter()
    converter.test_conversion() 