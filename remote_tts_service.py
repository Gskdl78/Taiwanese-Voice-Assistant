#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遠端 TTS 服務模組
連接到自訓練的 SuiSiann-HunLian TTS 系統
"""

import os
import time
import requests
import re
from urllib.parse import urlencode

class RemoteTtsService:
    """遠端 TTS 服務類別"""
    
    def __init__(self, remote_host="163.13.202.125", remote_port=5000):
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.base_url = f"http://{remote_host}:{remote_port}"
        self.endpoint = "/bangtsam"
        
        # 根據性能配置設定超時時間
        try:
            from performance_config import get_current_config
            config = get_current_config()
            self.timeout = config["tts_timeout"]
            print(f"使用優化的TTS超時設定: {self.timeout}秒")
        except ImportError:
            self.timeout = 45  # 預設45秒，因為要等標音轉換
            print(f"使用預設TTS超時設定: {self.timeout}秒")
        
    def generate_speech(self, text):
        """
        使用遠端TTS服務生成語音
        
        Args:
            text (str): 要合成的文字（數字調格式，如 "tak10-ke7 tsə2-hue1"）
            
        Returns:
            str: 生成的音檔路徑，如果失敗則返回 None
        """
        import time
        
        total_start = time.time()
        try:
            print(f"⏱️  遠端TTS開始: '{text}'")
            
            # 組合API URL和參數
            params_start = time.time()
            params = {"taibun": text}
            full_url = f"{self.base_url}{self.endpoint}"
            params_time = time.time() - params_start
            
            print(f"　├─ 參數準備耗時: {params_time:.3f}秒")
            print(f"　├─ 請求URL: {full_url}")
            print(f"　├─ 參數: {params}")
            
            # 發送請求到遠端TTS服務
            request_start = time.time()
            response = requests.get(
                full_url, 
                params=params, 
                timeout=self.timeout,
                headers={
                    'User-Agent': 'TaiwaneseVoiceChat/1.0',
                    'Accept': 'audio/wav, audio/*, */*',
                    'Accept-Language': 'zh-TW,zh;q=0.9,en;q=0.8'
                }
            )
            request_time = time.time() - request_start
            print(f"　├─ API請求耗時: {request_time:.3f}秒")
            
            print(f"　├─ 回應狀態碼: {response.status_code}")
            print(f"　├─ 回應大小: {len(response.content)} bytes")
            
            # 顯示回應標頭資訊，可能包含版本或其他資訊
            print("　├─ 回應標頭資訊:")
            for key, value in response.headers.items():
                print(f"　│   {key}: {value}")
            
            process_start = time.time()
            if response.status_code == 200:
                # 檢查回應是否為音檔格式
                content_type = response.headers.get('content-type', '').lower()
                is_audio = (
                    'audio' in content_type or 
                    len(response.content) > 1000 or 
                    response.content.startswith(b'RIFF') or 
                    response.content.startswith(b'ID3') or 
                    response.content.startswith(b'\xff\xfb')
                )
                
                if is_audio:
                    # 儲存音檔
                    save_start = time.time()
                    audio_file = self._save_audio_file(response.content, text)
                    save_time = time.time() - save_start
                    
                    if audio_file:
                        process_time = time.time() - process_start
                        total_time = time.time() - total_start
                        print(f"　├─ 音檔儲存耗時: {save_time:.3f}秒")
                        print(f"　├─ 處理總耗時: {process_time:.3f}秒")
                        print(f"✅ 遠端TTS成功，總耗時: {total_time:.3f}秒，音檔: {audio_file}")
                        return audio_file
                else:
                    total_time = time.time() - total_start
                    print(f"❌ 遠端TTS回應不是音檔格式，總耗時: {total_time:.3f}秒")
                    # 嘗試顯示錯誤訊息
                    try:
                        error_text = response.content.decode('utf-8', errors='ignore')[:200]
                        print(f"　├─ 錯誤內容: {error_text}")
                    except:
                        pass
            else:
                total_time = time.time() - total_start
                print(f"❌ 遠端TTS請求失敗，狀態碼: {response.status_code}，總耗時: {total_time:.3f}秒")
                
        except requests.exceptions.RequestException as e:
            total_time = time.time() - total_start
            print(f"❌ 遠端TTS連線錯誤: {e}，總耗時: {total_time:.3f}秒")
        except Exception as e:
            total_time = time.time() - total_start
            print(f"❌ 遠端TTS未知錯誤: {e}，總耗時: {total_time:.3f}秒")
            
        return None
    
    def test_with_params(self, text, additional_params=None):
        """
        使用額外參數測試TTS服務
        
        Args:
            text (str): 要合成的文字
            additional_params (dict): 額外的參數（如版本、語音模型等）
            
        Returns:
            str: 生成的音檔路徑，如果失敗則返回 None
        """
        try:
            print(f"🧪 測試TTS參數: '{text}'")
            
            # 基本參數
            params = {"taibun": text}
            
            # 加入額外參數
            if additional_params:
                params.update(additional_params)
                print(f"📝 額外參數: {additional_params}")
            
            full_url = f"{self.base_url}{self.endpoint}"
            
            print(f"請求URL: {full_url}")
            print(f"完整參數: {params}")
            
            # 發送請求
            response = requests.get(
                full_url, 
                params=params, 
                timeout=self.timeout,
                headers={
                    'User-Agent': 'TaiwaneseVoiceChat/1.0',
                    'Accept': 'audio/wav, audio/*, */*',
                    'Accept-Language': 'zh-TW,zh;q=0.9,en;q=0.8'
                }
            )
            
            print(f"回應狀態碼: {response.status_code}")
            print(f"回應大小: {len(response.content)} bytes")
            
            # 顯示回應標頭資訊
            print("📋 回應標頭資訊:")
            for key, value in response.headers.items():
                print(f"   {key}: {value}")
            
            if response.status_code == 200:
                # 檢查回應是否為音檔格式
                content_type = response.headers.get('content-type', '').lower()
                is_audio = (
                    'audio' in content_type or 
                    len(response.content) > 1000 or 
                    response.content.startswith(b'RIFF') or 
                    response.content.startswith(b'ID3') or 
                    response.content.startswith(b'\xff\xfb')
                )
                
                if is_audio:
                    # 儲存音檔，檔名包含參數資訊
                    param_info = "_".join([f"{k}={v}" for k, v in (additional_params or {}).items()])
                    filename_suffix = f"_{param_info}" if param_info else ""
                    
                    audio_file = self._save_audio_file_with_suffix(response.content, text, filename_suffix)
                    if audio_file:
                        print(f"✅ 測試成功，音檔儲存至: {audio_file}")
                        return audio_file
                else:
                    print("❌ 回應不是音檔格式")
                    try:
                        error_text = response.content.decode('utf-8', errors='ignore')[:200]
                        print(f"錯誤內容: {error_text}")
                    except:
                        pass
            else:
                print(f"❌ 請求失敗，狀態碼: {response.status_code}")
                
        except Exception as e:
            print(f"❌ 測試錯誤: {e}")
            
        return None
    
    def _save_audio_file(self, content, text_info=""):
        """
        儲存音檔到本地
        
        Args:
            content (bytes): 音檔內容
            text_info (str): 用於檔名的文字資訊
            
        Returns:
            str: 儲存的檔案路徑，如果失敗則返回 None
        """
        try:
            # 確保 static 目錄存在
            os.makedirs("static", exist_ok=True)
            
            # 生成安全的檔名
            timestamp = int(time.time())
            safe_text = re.sub(r'[<>:"/\\|?*,]', '_', text_info.replace(' ', '_'))[:20]
            filename = f"static/remote_tts_{safe_text}_{timestamp}.wav"
            
            # 寫入檔案
            with open(filename, 'wb') as f:
                f.write(content)
                
            print(f"音檔已儲存: {filename}")
            return filename
            
        except Exception as e:
            print(f"儲存音檔失敗: {e}")
            return None
    
    def _save_audio_file_with_suffix(self, content, text_info="", suffix=""):
        """
        儲存音檔到本地（帶有後綴）
        
        Args:
            content (bytes): 音檔內容
            text_info (str): 用於檔名的文字資訊
            suffix (str): 檔名後綴
            
        Returns:
            str: 儲存的檔案路徑，如果失敗則返回 None
        """
        try:
            # 確保 static 目錄存在
            os.makedirs("static", exist_ok=True)
            
            # 生成安全的檔名
            timestamp = int(time.time())
            safe_text = re.sub(r'[<>:"/\\|?*,]', '_', text_info.replace(' ', '_'))[:20]
            safe_suffix = re.sub(r'[<>:"/\\|?*,]', '_', suffix)
            filename = f"static/remote_tts_{safe_text}{safe_suffix}_{timestamp}.wav"
            
            # 寫入檔案
            with open(filename, 'wb') as f:
                f.write(content)
                
            print(f"音檔已儲存: {filename}")
            return filename
            
        except Exception as e:
            print(f"儲存音檔失敗: {e}")
            return None
    
    def test_connection(self):
        """
        測試與遠端TTS服務的連線
        
        Returns:
            bool: 連線成功返回 True，否則返回 False
        """
        try:
            # 使用簡單的測試文字
            test_text = "li2 ho2"  # "你好"
            result = self.generate_speech(test_text)
            return result is not None
        except Exception as e:
            print(f"連線測試失敗: {e}")
            return False 