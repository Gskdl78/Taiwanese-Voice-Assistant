#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
台語語音對話 Web 應用程式
整合台語 STT + 意傳科技 API（標音 + TTS）+ LLM 對話
基於 TauPhahJi-BangTsam 專案的 API 規範
"""

import os
import time
import tempfile
import subprocess
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
import librosa
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import torch
import requests
import json
import numpy as np
from urllib.parse import urlencode, quote

app = Flask(__name__)

# 全域變數
taiwanese_processor = None
taiwanese_model = None
device = None
ffmpeg_path = None
pipeline_asr = None

# 意傳科技 API 設定 (根據 TauPhahJi-BangTsam 文檔)
ITHUAN_API = {
    "標音服務": {
        "網域": "https://hokbu.ithuan.tw",
        "端點": "/tau",
        "方法": "POST",
        "內容類型": "application/x-www-form-urlencoded"
    },
    "整段語音合成": {
        "網域": "https://hokbu.ithuan.tw", 
        "端點": "/語音合成",
        "方法": "GET"
    },
    "單詞語音合成": {
        "網域": "https://hapsing.ithuan.tw",
        "端點": "/bangtsam", 
        "方法": "GET"
    }
}

# API使用限制 (根據文檔)
API_LIMITS = {
    "每分鐘下載限制": 3,  # 每IP每分鐘最多3次音檔下載
    "文字長度限制": 200,   # 建議單次查詢不超過200字
    "避免同時請求": True    # 避免同時發送多個請求
}

def find_ffmpeg():
    """尋找 FFmpeg 執行檔"""
    possible_paths = [
        "./ffmpeg_bin/ffmpeg.exe",  # 我們安裝的版本
        "ffmpeg",  # 系統 PATH 中
        "ffmpeg.exe",  # Windows 系統
        "C:/ffmpeg/bin/ffmpeg.exe",  # 常見安裝位置
    ]
    
    for path in possible_paths:
        try:
            result = subprocess.run([path, "-version"], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                return path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    
    return None

def init_taiwanese_model():
    """初始化台語語音辨識模型"""
    global taiwanese_processor, taiwanese_model, device, ffmpeg_path, pipeline_asr
    
    print("🎯 初始化台語語音辨識系統...")
    
    # 檢查設備
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  設備: {device}")
    
    # 檢查 FFmpeg
    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path:
        print(f"🎬 FFmpeg: {ffmpeg_path}")
    else:
        print("⚠️  FFmpeg: 未找到")
    
    # 台語專門模型清單（按優先順序）
    models_to_try = [
        "NUTN-KWS/Whisper-Taiwanese-model-v0.5",
        "EricChang/TAT-TD-openai-whisper-large-v2-Lora-epoch5-total5epoch",
        "cathyi/tw-tw-openai-whisper-large-v2-Lora-epoch5-total5epoch",
    ]
    
    print("🔄 載入台語專門模型...")
    
    # 嘗試載入台語模型
    for model_name in models_to_try:
        try:
            print(f"🔄 嘗試載入: {model_name}")
            
            # 方法1: 使用 transformers 直接載入
            try:
                processor = WhisperProcessor.from_pretrained(model_name)
                model = WhisperForConditionalGeneration.from_pretrained(model_name)
                model = model.to(device)
                
                taiwanese_processor = processor
                taiwanese_model = model
                print(f"✅ 成功載入台語模型: {model_name}")
                return True
                
            except Exception as e1:
                print(f"   方法1失敗: {e1}")
                
                # 方法2: 使用 pipeline
                try:
                    pipeline_asr = pipeline(
                        "automatic-speech-recognition",
                        model=model_name,
                        device=0 if device == "cuda" else -1,
                        return_timestamps=True
                    )
                    taiwanese_processor = "pipeline"
                    print(f"✅ 成功載入台語模型 (pipeline): {model_name}")
                    return True
                    
                except Exception as e2:
                    print(f"   方法2失敗: {e2}")
                    continue
                    
        except Exception as e:
            print(f"❌ 載入 {model_name} 失敗: {e}")
            continue
    
    # 如果台語模型都失敗，使用標準 Whisper
    print("⚠️  台語模型載入失敗，使用標準 Whisper")
    try:
        import whisper
        taiwanese_model = whisper.load_model("base")
        taiwanese_processor = "whisper_direct"
        print(f"✅ 標準 Whisper 模型載入成功！(設備: {device})")
        return True
        
    except Exception as e:
        print(f"❌ 所有模型載入失敗: {e}")
        taiwanese_processor = None
        taiwanese_model = None
        return False

def convert_webm_with_ffmpeg(webm_file):
    """使用 FFmpeg 將 webm 轉換為 wav"""
    global ffmpeg_path
    
    if not ffmpeg_path:
        print("   ❌ FFmpeg 不可用")
        return None
    
    try:
        print(f"   🔄 使用 FFmpeg 轉換: {ffmpeg_path}")
        
        # 創建臨時 wav 檔案
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_wav_path = temp_wav.name
        
        # 使用 FFmpeg 轉換
        cmd = [
            ffmpeg_path, '-i', webm_file, 
            '-ar', '16000',  # 設定取樣率
            '-ac', '1',      # 設定為單聲道
            '-y',            # 覆寫輸出檔案
            temp_wav_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"   ✅ FFmpeg 轉換成功")
            audio, sr = librosa.load(temp_wav_path, sr=16000)
            os.unlink(temp_wav_path)  # 刪除臨時檔案
            return audio, sr
        else:
            print(f"   ❌ FFmpeg 轉換失敗: {result.stderr}")
            if os.path.exists(temp_wav_path):
                os.unlink(temp_wav_path)
            return None
            
    except Exception as e:
        print(f"   ❌ FFmpeg 轉換出錯: {e}")
        return None

def transcribe_taiwanese_audio(audio_file_path):
    """台語語音辨識（支援多種格式）"""
    global taiwanese_processor, taiwanese_model, pipeline_asr, device
    
    try:
        print(f"🎤 開始台語語音辨識: {audio_file_path}")
        
        # 檢查檔案
        if not os.path.exists(audio_file_path):
            print("❌ 音檔不存在")
            return ""
        
        # 根據檔案類型處理音檔
        print(f"📁 音檔路徑: {audio_file_path}")
        print(f"📏 檔案大小: {os.path.getsize(audio_file_path)} bytes")
        
        # 嘗試載入音檔
        audio_data = None
        sr = 16000
        
        # 處理 webm 格式
        if audio_file_path.lower().endswith('.webm'):
            print("🔄 處理 WebM 格式...")
            result = convert_webm_with_ffmpeg(audio_file_path)
            if result:
                audio_data, sr = result
            else:
                print("❌ WebM 轉換失敗")
                return ""
        else:
            # 處理其他格式
            try:
                print("🔄 直接載入音檔...")
                audio_data, sr = librosa.load(audio_file_path, sr=16000)
                print(f"   ✅ 載入成功")
            except Exception as e:
                print(f"   ❌ 載入失敗: {e}")
                return ""
        
        # 檢查音檔資料
        if audio_data is None or len(audio_data) == 0:
            print("❌ 音檔資料為空")
            return ""
        
        print(f"📊 音檔資訊: 長度={len(audio_data)}, 取樣率={sr}")
        
        transcription = ""
        
        # 方法1: 使用台語 transformers 模型
        if taiwanese_processor and taiwanese_model and taiwanese_processor != "pipeline" and taiwanese_processor != "whisper_direct":
            try:
                print("🔄 使用台語 transformers 模型...")
                
                # 準備輸入
                inputs = taiwanese_processor(audio_data, sampling_rate=sr, return_tensors="pt")
                inputs = inputs.to(device)
                
                # 生成轉錄
                with torch.no_grad():
                    predicted_ids = taiwanese_model.generate(inputs["input_features"])
                    transcription = taiwanese_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
                print(f"✅ transformers 辨識成功: '{transcription}'")
                
            except Exception as e:
                print(f"❌ transformers 模型失敗: {e}")
        
        # 方法2: 使用 pipeline
        if not transcription and pipeline_asr and taiwanese_processor == "pipeline":
            try:
                print("🔄 使用 pipeline 模型...")
                
                # 暫存音檔
                temp_wav = f"temp_{int(time.time())}.wav"
                sf.write(temp_wav, audio_data, sr)
                
                result = pipeline_asr(temp_wav)
                transcription = result["text"]
                
                # 清理暫存檔
                if os.path.exists(temp_wav):
                    os.unlink(temp_wav)
                
                print(f"✅ pipeline 辨識成功: '{transcription}'")
                
            except Exception as e:
                print(f"❌ pipeline 模型失敗: {e}")
        
        # 方法3: 使用標準 Whisper
        if not transcription and taiwanese_model and taiwanese_processor == "whisper_direct":
            try:
                print("🔄 使用標準 Whisper 模型...")
                
                # 暫存音檔
                temp_wav = f"temp_{int(time.time())}.wav"
                sf.write(temp_wav, audio_data, sr)
                
                result = taiwanese_model.transcribe(temp_wav, language="zh")
                transcription = result["text"]
                
                # 清理暫存檔
                if os.path.exists(temp_wav):
                    os.unlink(temp_wav)
                
                print(f"✅ Whisper 辨識成功: '{transcription}'")
                
            except Exception as e:
                print(f"❌ Whisper 模型失敗: {e}")
        
        # 清理結果
        if transcription:
            transcription = transcription.strip()
            print(f"🎯 最終辨識結果: '{transcription}'")
            return transcription
        else:
            print("❌ 所有辨識方法都失敗")
            return ""
            
    except Exception as e:
        print(f"❌ 語音辨識出錯: {e}")
        import traceback
        traceback.print_exc()
        return ""

def get_taiwanese_pronunciation(text):
    """調用意傳科技標音 API，獲取台語羅馬拼音（按照 TauPhahJi-BangTsam 規範）"""
    try:
        print(f"🔤 獲取台語標音: '{text}'")
        
        # 檢查文字長度限制
        if len(text) > API_LIMITS["文字長度限制"]:
            print(f"⚠️ 文字過長 ({len(text)} > {API_LIMITS['文字長度限制']})，截斷處理")
            text = text[:API_LIMITS["文字長度限制"]]
        
        # 根據文檔規範構建API請求
        api_config = ITHUAN_API["標音服務"]
        url = f"{api_config['網域']}{api_config['端點']}"
        
        # POST 請求，參數在 body 中（application/x-www-form-urlencoded）
        data = {
            'taibun': text.strip()
        }
        
        print(f"   📡 API 端點: {url}")
        print(f"   📤 請求參數: {data}")
        print(f"   🔧 請求方法: {api_config['方法']}")
        print(f"   📋 內容類型: {api_config['內容類型']}")
        
        # 發送請求
        response = requests.post(
            url,
            data=data,
            headers={
                'Content-Type': api_config['內容類型'],
                'User-Agent': 'TaiwaneseVoiceChat/1.0'
            },
            timeout=15
        )
        
        print(f"   📥 回應狀態: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   📋 標音結果: {result}")
            
            # 根據文檔格式解析結果
            if 'kiatko' in result and result['kiatko']:
                # 組合羅馬拼音
                romanization_parts = []
                for item in result['kiatko']:
                    if 'KIP' in item and item['KIP']:
                        romanization_parts.append(item['KIP'])
                
                if romanization_parts:
                    romanization = ' '.join(romanization_parts)
                    print(f"   ✅ 羅馬拼音: {romanization}")
                    return romanization, result.get('分詞', text), result['kiatko']
            
            # 如果沒有 kiatko，使用分詞結果
            if '分詞' in result:
                segmented = result['分詞']
                print(f"   📝 分詞結果: {segmented}")
                return segmented, segmented, []
        
        print(f"   ⚠️ API 返回異常: {response.text[:200]}")
        return text, text, []
        
    except Exception as e:
        print(f"   ❌ 標音 API 失敗: {e}")
        return text, text, []

def text_to_speech_ithuan_full_sentence(腔口, 分詞):
    """整段語音合成（按照 TauPhahJi-BangTsam 規範）"""
    try:
        print(f"🔊 整段語音合成: 腔口='{腔口}', 分詞='{分詞}'")
        
        # 根據文檔規範構建API請求
        api_config = ITHUAN_API["整段語音合成"]
        base_url = f"{api_config['網域']}{api_config['端點']}"
        
        # GET 請求參數
        params = {
            '查詢腔口': 腔口,
            '查詢語句': 分詞
        }
        
        print(f"   📡 API 端點: {base_url}")
        print(f"   📤 請求參數: {params}")
        print(f"   🔧 請求方法: {api_config['方法']}")
        
        # 手動構建 URL 以確保正確編碼（按照文檔範例）
        encoded_params = urlencode(params, safe='', encoding='utf-8')
        full_url = f"{base_url}?{encoded_params}"
        print(f"   🌐 完整URL: {full_url}")
        
        # 發送 GET 請求
        response = requests.get(
            full_url,
            timeout=30,
            headers={
                'User-Agent': 'TaiwaneseVoiceChat/1.0',
                'Accept': 'audio/wav, audio/*, */*',
                'Accept-Language': 'zh-TW,zh;q=0.9,en;q=0.8'
            }
        )
        
        print(f"   📥 回應狀態: {response.status_code}")
        print(f"   📋 Content-Type: {response.headers.get('content-type', 'unknown')}")
        print(f"   📏 回應大小: {len(response.content)} bytes")
        
        # 不管狀態碼如何，都檢查是否有音檔數據
        if len(response.content) > 100:
            content_type = response.headers.get('content-type', '').lower()
            print(f"   🔍 檢查音檔內容: 大小={len(response.content)}, type='{content_type}'")
            
            # 檢查是否為音訊內容（更寬鬆的判斷）
            is_audio = (
                'audio' in content_type or 
                len(response.content) > 1000 or
                response.content.startswith(b'RIFF') or  # WAV檔案標頭
                response.content.startswith(b'ID3') or   # MP3檔案標頭
                response.content.startswith(b'\xff\xfb') # MP3檔案標頭變種
            )
            
            if is_audio:
                # 儲存音檔（即使是404也嘗試保存）
                timestamp = int(time.time())
                output_file = f"static/tts_full_{timestamp}.wav"
                
                # 確保 static 目錄存在
                os.makedirs("static", exist_ok=True)
                
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                
                print(f"   ✅ 發現音檔數據並保存: {output_file} (狀態碼: {response.status_code})")
                return output_file
            else:
                print(f"   ⚠️ 回應不像音訊格式")
                # 如果是文字回應，顯示內容
                if response.content:
                    try:
                        text_content = response.content.decode('utf-8', errors='ignore')[:200]
                        print(f"   📋 文字回應: {text_content}")
                    except:
                        print(f"   📋 二進位回應: {response.content[:50]}")
        else:
            print(f"   ⚠️ 回應太小（{len(response.content)} bytes），可能不是音檔")
            if response.content:
                try:
                    text_content = response.content.decode('utf-8', errors='ignore')[:200]
                    print(f"   📋 小回應內容: {text_content}")
                except:
                    print(f"   📋 小回應二進位: {response.content}")
        
        return None
        
    except Exception as e:
        print(f"   ❌ 整段語音合成失敗: {e}")
        return None

def text_to_speech_ithuan_single_word(羅馬拼音):
    """單詞語音合成（按照 TauPhahJi-BangTsam 規範）"""
    try:
        print(f"🔊 單詞語音合成: '{羅馬拼音}'")
        
        # 根據文檔規範構建API請求
        api_config = ITHUAN_API["單詞語音合成"]
        base_url = f"{api_config['網域']}{api_config['端點']}"
        
        # GET 請求參數
        params = {
            'taibun': 羅馬拼音
        }
        
        print(f"   📡 API 端點: {base_url}")
        print(f"   📤 請求參數: {params}")
        print(f"   🔧 請求方法: {api_config['方法']}")
        
        # 按照文檔範例構建URL
        # encodeURI + encodeURIComponent
        encoded_taibun = quote(羅馬拼音, safe='')
        full_url = f"{base_url}?taibun={encoded_taibun}"
        print(f"   🌐 完整URL: {full_url}")
        
        # 發送 GET 請求
        response = requests.get(
            full_url,
            timeout=30,
            headers={
                'User-Agent': 'TaiwaneseVoiceChat/1.0',
                'Accept': 'audio/wav, audio/*, */*',
                'Accept-Language': 'zh-TW,zh;q=0.9,en;q=0.8'
            }
        )
        
        print(f"   📥 回應狀態: {response.status_code}")
        print(f"   📋 Content-Type: {response.headers.get('content-type', 'unknown')}")
        print(f"   📏 回應大小: {len(response.content)} bytes")
        
        # 不管狀態碼如何，都檢查是否有音檔數據
        if len(response.content) > 100:
            content_type = response.headers.get('content-type', '').lower()
            print(f"   🔍 檢查音檔內容: 大小={len(response.content)}, type='{content_type}'")
            
            # 檢查是否為音訊內容（更寬鬆的判斷）
            is_audio = (
                'audio' in content_type or 
                len(response.content) > 1000 or
                response.content.startswith(b'RIFF') or  # WAV檔案標頭
                response.content.startswith(b'ID3') or   # MP3檔案標頭
                response.content.startswith(b'\xff\xfb') # MP3檔案標頭變種
            )
            
            if is_audio:
                # 儲存音檔（即使是404也嘗試保存）
                timestamp = int(time.time())
                # 清理檔名，移除不合法字符
                import re
                safe_filename = re.sub(r'[<>:"/\\|?*,]', '_', 羅馬拼音.replace(' ', '_').replace('-', '_'))[:20]
                output_file = f"static/tts_word_{safe_filename}_{timestamp}.wav"
                
                # 確保 static 目錄存在
                os.makedirs("static", exist_ok=True)
                
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                
                print(f"   ✅ 發現音檔數據並保存: {output_file} (狀態碼: {response.status_code})")
                return output_file
            else:
                print(f"   ⚠️ 回應不像音訊格式")
                # 如果是文字回應，顯示內容
                if response.content:
                    try:
                        text_content = response.content.decode('utf-8', errors='ignore')[:200]
                        print(f"   📋 文字回應: {text_content}")
                    except:
                        print(f"   📋 二進位回應: {response.content[:50]}")
        else:
            print(f"   ⚠️ 回應太小（{len(response.content)} bytes），可能不是音檔")
            if response.content:
                try:
                    text_content = response.content.decode('utf-8', errors='ignore')[:200]
                    print(f"   📋 小回應內容: {text_content}")
                except:
                    print(f"   📋 小回應二進位: {response.content}")
        
        return None
        
    except Exception as e:
        print(f"   ❌ 單詞語音合成失敗: {e}")
        return None

def chat_with_ollama(text):
    """與 Ollama LLM 對話，返回台語回應"""
    try:
        print(f"🤖 LLM 對話處理: '{text}'")
        
        # 構建提示詞，要求簡短的台語回應
        prompt = f"""你是一個親切的台語助手。請用台語漢字簡短回應以下話語：

規則：
- 只能回應 3-8 個字
- 使用常見的繁體中文漢字
- 語氣要親切自然
- 不要重複同一個字超過 2 次
- 要符合台語的說話習慣

用戶說：{text}

台語回應："""

        # 調用 Ollama API
        response = requests.post('http://localhost:11434/api/generate', 
            json={
                'model': 'gemma3:4b',
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.6,
                    'num_predict': 30,  # 進一步限制長度
                    'top_p': 0.9
                }
            },
            timeout=20
        )
        
        if response.status_code == 200:
            result = response.json()
            reply = result.get('response', '').strip()
            
            # 清理回應，只保留中文漢字和基本標點
            import re
            cleaned_reply = re.sub(r'[^\u4e00-\u9fff！？。，、]', '', reply)
            
            # 移除重複超過 2 次的字符
            def remove_excessive_repeats(text):
                result = ""
                char_count = {}
                for char in text:
                    char_count[char] = char_count.get(char, 0) + 1
                    if char_count[char] <= 2:  # 允許最多重複 2 次
                        result += char
                return result
            
            cleaned_reply = remove_excessive_repeats(cleaned_reply)
            
            # 確保回應在8字內且不為空
            if len(cleaned_reply) > 8:
                cleaned_reply = cleaned_reply[:8]
            
            # 如果清理後為空或太短，使用預設回應
            if len(cleaned_reply) < 2:
                final_reply = "好欸！"
            else:
                final_reply = cleaned_reply
                
            print(f"🤖 LLM 回應: '{final_reply}'")
            return final_reply
        else:
            print(f"⚠️  LLM API 失敗: {response.status_code}")
            return "好欸！"
            
    except Exception as e:
        print(f"❌ LLM 對話失敗: {e}")
        return "好欸！"

def text_to_speech_ithuan(text, kiatko_data=None):
    """意傳科技 TTS 主函數（整合整段和單詞合成）"""
    print(f"🔊 意傳科技 TTS 開始: '{text}'")
    
    # 優先嘗試整段語音合成
    print("🎯 嘗試整段語音合成...")
    full_audio = text_to_speech_ithuan_full_sentence("閩南語", text)
    if full_audio:
        print(f"✅ 整段語音合成成功: {full_audio}")
        return full_audio
    
    # 如果整段失敗，且有分詞資料，嘗試合成第一個詞
    if kiatko_data and len(kiatko_data) > 0:
        print("🎯 嘗試單詞語音合成...")
        first_word = kiatko_data[0]
        if 'KIP' in first_word and first_word['KIP']:
            single_audio = text_to_speech_ithuan_single_word(first_word['KIP'])
            if single_audio:
                print(f"✅ 單詞語音合成成功: {single_audio}")
                return single_audio
    
    print("❌ 所有意傳科技 TTS 方案都失敗")
    return None

@app.route('/')
def index():
    """主頁面"""
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """處理語音檔案（支援 webm 和其他格式）"""
    try:
        # 獲取上傳的音檔
        if 'audio' not in request.files:
            print("❌ 沒有收到音檔")
            return jsonify({'error': '沒有收到音檔'}), 400
        
        audio_file = request.files['audio']
        
        # 詳細檢查音檔資訊
        content_type = getattr(audio_file, 'content_type', 'unknown')
        filename = getattr(audio_file, 'filename', 'unknown')
        
        print(f"📥 收到音檔: {filename}")
        print(f"   檔案類型: {content_type}")
        
        # 檢查音檔是否為空
        if filename == '':
            print("❌ 音檔名稱為空")
            return jsonify({'error': '音檔名稱為空'}), 400
            
        # 建立本地保存目錄
        os.makedirs("uploads", exist_ok=True)
        
        # 決定檔案副檔名
        if 'webm' in content_type:
            suffix = '.webm'
        elif 'wav' in content_type:
            suffix = '.wav'
        elif 'mp3' in content_type:
            suffix = '.mp3'
        else:
            suffix = '.audio'
            
        print(f"   檔案類型: {content_type} -> 副檔名: {suffix}")
        
        # 建立本地檔案路徑
        timestamp = int(time.time() * 1000)  # 毫秒時間戳
        local_filename = f"uploads/recording_{timestamp}{suffix}"
        
        print(f"💾 保存音檔到: {local_filename}")
        
        # 讀取並保存音檔數據
        audio_file.seek(0)
        audio_data = audio_file.read()
        actual_size = len(audio_data)
        print(f"   音檔大小: {actual_size} bytes")
        
        if actual_size == 0:
            print("❌ 音檔數據為空")
            return jsonify({'error': '音檔數據為空'}), 400
        
        # 寫入本地檔案
        try:
            with open(local_filename, 'wb') as f:
                f.write(audio_data)
            print(f"✅ 音檔保存成功")
        except Exception as e:
            print(f"❌ 音檔保存失敗: {e}")
            return jsonify({'error': '音檔保存失敗'}), 500
        
        # 驗證保存的檔案
        if os.path.exists(local_filename):
            saved_size = os.path.getsize(local_filename)
            print(f"📁 本地檔案大小: {saved_size} bytes")
            if saved_size == 0:
                print("❌ 保存的音檔為空")
                return jsonify({'error': '保存的音檔為空'}), 400
        else:
            print("❌ 音檔保存失敗")
            return jsonify({'error': '音檔保存失敗'}), 500
        
        try:
            print("="*60)
            print("🚀 開始台語語音對話處理（基於 TauPhahJi-BangTsam 規範）")
            print("="*60)
            
            # 1. 台語語音辨識
            print("📝 步驟1: 台語語音辨識")
            recognized_text = transcribe_taiwanese_audio(local_filename)
            print(f"📝 辨識結果: '{recognized_text}'")
            
            if not recognized_text:
                print("❌ 辨識失敗，返回錯誤")
                return jsonify({'error': '無法辨識台語語音內容'}), 400
            
            # 2. LLM 對話（直接用中文辨識結果）
            print("\n🤖 步驟2: AI 生成回應")
            ai_response = chat_with_ollama(recognized_text)
            print(f"🤖 AI 回應: '{ai_response}'")
            
            # 3. 將 AI 回應轉換為台語標音（用於 TTS）
            print("\n🔤 步驟3: AI 回應台語標音轉換")
            romanization, segmented, kiatko_data = get_taiwanese_pronunciation(ai_response)
            print(f"🔤 羅馬拼音: '{romanization}'")
            print(f"📝 分詞結果: '{segmented}'")
            print(f"📊 分詞資料筆數: {len(kiatko_data)}")
            
            # 4. 文字轉語音（使用意傳科技 TTS）
            print("\n🔊 步驟4: 台語語音合成")
            print(f"⚠️ API 限制提醒: 每IP每分鐘最多{API_LIMITS['每分鐘下載限制']}次音檔下載")
            audio_file_path = text_to_speech_ithuan(romanization, kiatko_data)
            
            if audio_file_path:
                print(f"🔊 TTS 成功: {audio_file_path}")
            else:
                print("⚠️ TTS 失敗")
            
            # 5. 返回結果
            print("\n✅ 台語語音對話處理完成")
            result = {
                'recognized_text': recognized_text,
                'ai_response': ai_response,
                'romanization': romanization,
                'segmented': segmented,
                'kiatko_count': len(kiatko_data),
                'audio_url': f'/{audio_file_path}' if audio_file_path else None,
                'api_info': f"使用意傳科技 API (限制: {API_LIMITS['每分鐘下載限制']}次/分鐘)"
            }
            
            print("="*60)
            return jsonify(result)
            
        finally:
            # 可選擇性清理本地檔案
            CLEANUP_FILES = False  # 保留檔案以便除錯
            
            if CLEANUP_FILES and os.path.exists(local_filename):
                try:
                    os.unlink(local_filename)
                    print(f"🗑️ 清理本地檔案: {local_filename}")
                except Exception as e:
                    print(f"⚠️ 清理檔案失敗: {e}")
            else:
                print(f"📁 本地檔案保留: {local_filename}")
        
    except Exception as e:
        print(f"❌ 處理錯誤: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'處理失敗: {str(e)}'}), 500

@app.route('/test_api')
def test_api():
    """測試意傳科技 API（按照 TauPhahJi-BangTsam 規範）"""
    try:
        test_text = "你好嗎"
        print(f"🧪 測試意傳科技 API: '{test_text}'")
        print(f"📋 API 規範: 基於 TauPhahJi-BangTsam 專案")
        
        # 測試標音 API
        romanization, segmented, kiatko_data = get_taiwanese_pronunciation(test_text)
        print(f"🔤 測試標音結果: '{romanization}'")
        
        # 測試 TTS API
        print(f"🔊 測試 TTS 輸入: '{romanization}' (羅馬拼音)")
        audio_file = text_to_speech_ithuan(romanization, kiatko_data)
        
        result = {
            'test_text': test_text,
            'romanization': romanization,
            'segmented': segmented,
            'kiatko_count': len(kiatko_data),
            'audio_file': audio_file,
            'api_status': 'success' if audio_file else 'partial',
            'api_info': {
                '標音服務': f"{ITHUAN_API['標音服務']['網域']}{ITHUAN_API['標音服務']['端點']}",
                '整段語音': f"{ITHUAN_API['整段語音合成']['網域']}{ITHUAN_API['整段語音合成']['端點']}",
                '單詞語音': f"{ITHUAN_API['單詞語音合成']['網域']}{ITHUAN_API['單詞語音合成']['端點']}",
                '使用限制': f"每IP每分鐘最多{API_LIMITS['每分鐘下載限制']}次下載"
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e), 'api_status': 'failed'})

@app.route('/static/<path:filename>')
def serve_static(filename):
    """提供靜態檔案"""
    return send_file(f'static/{filename}')

if __name__ == '__main__':
    print("🎯 啟動台語語音對話 Web 應用程式")
    print("🌐 整合意傳科技 API（基於 TauPhahJi-BangTsam 規範）")
    print("🔧 已修復404錯誤音檔捕獲問題")
    print("📚 API 文檔來源: TauPhahJi-API-docs/API和組件文檔.md")
    print("=" * 50)
    
    # 顯示API配置資訊
    print("🔧 API 配置資訊:")
    for service_name, config in ITHUAN_API.items():
        print(f"   {service_name}: {config['網域']}{config['端點']} ({config['方法']})")
    
    print(f"⚠️ 使用限制: 每IP每分鐘最多{API_LIMITS['每分鐘下載限制']}次音檔下載")
    print("=" * 50)
    
    # 初始化台語模型
    model_ready = init_taiwanese_model()
    
    if model_ready:
        print("✅ 系統初始化完成！")
        print(f"🌐 啟動 Web 服務...")
        print(f"📱 訪問 http://localhost:5000 開始使用")
        print(f"🧪 訪問 http://localhost:5000/test_api 測試 API")
        print("=" * 50)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ 系統初始化失敗，無法啟動 Web 服務")
        print("請檢查模型安裝和相關依賴") 