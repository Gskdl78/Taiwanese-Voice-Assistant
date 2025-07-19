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
import re
from flask import Flask, render_template, request, jsonify, send_file
import librosa
import soundfile as sf
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import requests
from urllib.parse import urlencode, quote

# 匯入新的TTS服務和格式轉換器
from remote_tts_service import RemoteTtsService
from romanization_converter import RomanizationConverter

# 匯入性能配置
try:
    from performance_config import (
        get_current_config, get_optimization_suggestions,
        apply_performance_mode, PERFORMANCE_MODE
    )
    PERFORMANCE_CONFIG_AVAILABLE = True
except ImportError:
    print("⚠️ 性能配置模組未找到，使用預設配置")
    PERFORMANCE_CONFIG_AVAILABLE = False
    PERFORMANCE_MODE = "fast"

app = Flask(__name__)

# 全域變數
CLEANUP_FILES = True  # 是否清理臨時檔案
AUDIO_SAMPLE_RATE = 16000  # 音訊取樣率
MAX_AUDIO_DURATION = 30  # 最長音訊時間（秒）
MIN_AUDIO_DURATION = 0.5  # 最短音訊時間（秒）

def debug_print(message):
    """調試輸出函數"""
    print(f"[DEBUG] {message}")

def performance_timer(func_name):
    """性能計時裝飾器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            print(f"⏱️  開始執行: {func_name}")
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                print(f"✅ {func_name} 完成 - 耗時: {duration:.3f}秒")
                return result
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                print(f"❌ {func_name} 失敗 - 耗時: {duration:.3f}秒 - 錯誤: {e}")
                raise
                
        return wrapper
    return decorator

def log_step_time(step_name, duration, details=""):
    """記錄步驟執行時間"""
    print(f"📊 【{step_name}】耗時: {duration:.3f}秒 {details}")

# 全域變數
taiwanese_processor = None
taiwanese_model = None
device = None
ffmpeg_path = None
pipeline_asr = None
remote_tts_service = None
romanization_converter = None

# 音訊預處理參數
AUDIO_PREPROCESSING = {
    "remove_silence": True,  # 移除靜音
    "noise_reduction": True,  # 降噪
    "normalize_volume": True,  # 音量正規化
    "silence_threshold": 0.05,  # 靜音閾值
    "min_silence_duration": 0.3,  # 最小靜音時間（秒）
}

def preprocess_audio(audio_path):
    """
    音訊預處理函數（使用 GPU 加速）
    
    Args:
        audio_path (str): 輸入音訊檔案路徑
        
    Returns:
        str: 處理後的音訊檔案路徑
    """
    try:
        print(f"🎵 開始音訊預處理: {audio_path}")
        
        # 使用 torchaudio 讀取音訊（支援 GPU 加速）
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 確保單聲道
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 重採樣到目標採樣率
        if sample_rate != AUDIO_SAMPLE_RATE:
            resampler = T.Resample(sample_rate, AUDIO_SAMPLE_RATE)
            waveform = resampler(waveform)
        
        # 移動到 GPU（如果可用）
        waveform = waveform.to(device)
        
        if AUDIO_PREPROCESSING["remove_silence"]:
            # 使用 torchaudio 的靜音檢測
            db_threshold = 20
            frame_length = 2048
            hop_length = 512
            
            # 計算能量
            energy = torch.norm(waveform.view(-1, frame_length), dim=1)
            energy_db = 20 * torch.log10(energy + 1e-10)
            
            # 找出非靜音段落
            mask = energy_db > -db_threshold
            mask = mask.to(device)
            
            # 應用遮罩
            waveform = waveform.squeeze()
            non_silent = waveform[mask]
            waveform = non_silent.unsqueeze(0)
        
        if AUDIO_PREPROCESSING["noise_reduction"]:
            # 使用 GPU 加速的頻譜處理
            n_fft = 2048
            hop_length = 512
            
            # 計算 STFT
            spec = torch.stft(
                waveform.squeeze(),
                n_fft=n_fft,
                hop_length=hop_length,
                window=torch.hann_window(n_fft).to(device),
                return_complex=True
            )
            
            # 計算頻譜幅度
            spec_mag = torch.abs(spec)
            
            # 估計噪音頻譜
            noise_estimate = torch.mean(spec_mag[:, :10], dim=1, keepdim=True)
            
            # 頻譜減法
            spec_mag_clean = torch.maximum(spec_mag - noise_estimate, torch.zeros_like(spec_mag))
            
            # 重建相位
            phase = torch.angle(spec)
            spec_clean = spec_mag_clean * torch.exp(1j * phase)
            
            # 反轉 STFT
            waveform = torch.istft(
                spec_clean,
                n_fft=n_fft,
                hop_length=hop_length,
                window=torch.hann_window(n_fft).to(device),
                length=waveform.size(-1)
            )
            waveform = waveform.unsqueeze(0)
        
        if AUDIO_PREPROCESSING["normalize_volume"]:
            # 音量正規化
            waveform = waveform / torch.max(torch.abs(waveform))
        
        # 移回 CPU 並儲存
        waveform = waveform.cpu()
        output_path = audio_path.replace('.wav', '_processed.wav')
        torchaudio.save(output_path, waveform, AUDIO_SAMPLE_RATE)
        
        print(f"✅ 音訊預處理完成: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ 音訊預處理失敗: {e}")
        return audio_path

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
        "端點": "/bangtsam",
        "方法": "GET",
        "內容類型": "application/x-www-form-urlencoded"
    },
    "單詞語音合成": {
        "網域": "https://hokbu.ithuan.tw",
        "端點": "/huan",
        "方法": "GET",
        "內容類型": "application/x-www-form-urlencoded"
    }
}

# API使用限制
API_LIMITS = {
    "每分鐘下載限制": 3,  # 每IP每分鐘最多3次音檔下載
    "文字長度限制": 200,   # 建議單次查詢不超過200字
    "避免同時請求": True    # 避免同時發送多個請求
}

def find_ffmpeg():
    """尋找 FFmpeg 執行檔"""
    possible_paths = [
        "./ffmpeg_bin/ffmpeg.exe",
        "ffmpeg",
        "ffmpeg.exe",
        "C:/ffmpeg/bin/ffmpeg.exe",
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
    global taiwanese_processor, taiwanese_model, device, ffmpeg_path
    
    debug_print("初始化台語語音辨識系統...")
    
    # 檢查設備
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"🎮 使用 GPU 加速: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # 啟用 CUDA 優化
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        print("⚠️ 未檢測到 GPU，使用 CPU 模式")
    
    debug_print(f"設備: {device}")
    
    # 檢查 FFmpeg
    ffmpeg_path = find_ffmpeg()
    debug_print(f"FFmpeg: {ffmpeg_path if ffmpeg_path else '未找到'}")
    
    # 台語專門模型
    model_name = "NUTN-KWS/Whisper-Taiwanese-model-v0.5"
    
    debug_print("載入台語專門模型...")
    debug_print(f"嘗試載入: {model_name}")
    
    try:
        # 使用 float16 和優化配置
        load_config = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "low_cpu_mem_usage": True,
            "use_safetensors": True
        }
        
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            **load_config
        )
        
        # 移動到 GPU 並優化
        model = model.to(device)
        if device == "cuda":
            model.eval()  # 設定為評估模式
        
        taiwanese_processor = processor
        taiwanese_model = model
        
        debug_print(f"✅ 成功載入台語模型: {model_name}")
        return True
            
    except Exception as e:
        debug_print(f"載入台語模型失敗: {e}")
        taiwanese_processor = None
        taiwanese_model = None
        return False

def convert_webm_with_ffmpeg(webm_file):
    """使用 FFmpeg 將 webm 轉換為 wav"""
    global ffmpeg_path
    
    if not ffmpeg_path:
        debug_print("FFmpeg 不可用")
        return None
    
    try:
        debug_print(f"使用 FFmpeg 轉換: {webm_file}")
        
        # 檢查輸入檔案
        if not os.path.exists(webm_file):
            debug_print(f"輸入檔案不存在: {webm_file}")
            return None
            
        # 建立臨時檔案
        temp_wav = os.path.join(
            os.path.dirname(webm_file),
            f"temp_{int(time.time()*1000)}.wav"
        )
        
        # FFmpeg 命令
        cmd = [
            ffmpeg_path,
            '-y',  # 覆寫輸出檔案
            '-i', webm_file,  # 輸入
            '-acodec', 'pcm_s16le',  # 音訊編碼
            '-ar', '16000',  # 採樣率
            '-ac', '1',  # 單聲道
            '-hide_banner',  # 隱藏橫幅
            '-loglevel', 'error',  # 只顯示錯誤
            temp_wav  # 輸出
        ]
        
        debug_print(f"執行命令: {' '.join(cmd)}")
        
        # 執行轉換
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            if os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 0:
                debug_print(f"FFmpeg 轉換成功: {temp_wav}")
                return temp_wav
            else:
                debug_print("轉換後的檔案無效")
                return None
        else:
            debug_print(f"FFmpeg 轉換失敗: {result.stderr}")
            return None
            
    except Exception as e:
        debug_print(f"FFmpeg 轉換出錯: {e}")
        return None

def transcribe_taiwanese_audio(audio_file_path):
    """台語語音辨識"""
    global taiwanese_processor, taiwanese_model, device
    
    try:
        debug_print(f"開始台語語音辨識: {audio_file_path}")
        
        if not os.path.exists(audio_file_path):
            debug_print("音檔不存在")
            return ""
        
        # 載入音檔
        audio_data = None
        sr = 16000
        
        if audio_file_path.lower().endswith('.webm'):
            debug_print("處理 WebM 格式...")
            converted_path = convert_webm_with_ffmpeg(audio_file_path)
            if converted_path:
                try:
                    audio_data, sr = librosa.load(converted_path, sr=16000, mono=True)
                    os.unlink(converted_path)  # 清理臨時檔案
                    debug_print("WebM 轉換成功")
                except Exception as e:
                    debug_print(f"WebM 音檔載入失敗: {e}")
                    if os.path.exists(converted_path):
                        os.unlink(converted_path)
                    return ""
            else:
                debug_print("WebM 轉換失敗")
                return ""
        else:
            try:
                audio_data, sr = librosa.load(audio_file_path, sr=16000, mono=True)
                debug_print("音檔載入成功")
            except Exception as e:
                debug_print(f"音檔載入失敗: {e}")
                return ""
        
        if audio_data is None or len(audio_data) == 0:
            debug_print("音檔資料為空")
            return ""
        
        debug_print(f"音檔資訊: 長度={len(audio_data)}, 取樣率={sr}")
        
        # 使用台語 transformers 模型
        try:
            debug_print("使用台語 transformers 模型...")
            
            # 計算特徵
            input_features = taiwanese_processor.feature_extractor(
                audio_data, 
                sampling_rate=sr,
                return_tensors="pt"
            )
            
            # 移動到 GPU
            input_features = input_features.to(device)
            
            # 使用混合精度推理
            with torch.cuda.amp.autocast():
                generated_ids = taiwanese_model.generate(
                    input_features.input_features,
                    max_length=225,
                    language="zh",
                    task="transcribe",
                    num_beams=1,
                    do_sample=False
                )
                
                transcription = taiwanese_processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0].strip()
            
            # 清理 GPU 記憶體
            if device == "cuda":
                torch.cuda.empty_cache()
            
            debug_print(f"辨識成功: '{transcription}'")
            return transcription
            
        except Exception as e:
            debug_print(f"辨識失敗: {e}")
            return ""
            
    except Exception as e:
        debug_print(f"語音辨識出錯: {e}")
        return ""

def clean_transcription_result(text):
    """清理辨識結果文字"""
    if not text:
        return text
        
    # 移除多餘的空格
    text = re.sub(r'\s+', ' ', text)
    
    # 移除特殊標記
    text = re.sub(r'<[^>]+>', '', text)
    
    # 修正常見錯誤
    text = text.replace('台語:', '')
    text = text.replace('台羅:', '')
    
    return text.strip()

@performance_timer("台語標音轉換")
def get_taiwanese_pronunciation(text):
    """調用意傳科技標音 API"""
    try:
        debug_print(f"獲取台語標音: '{text}'")
        
        if len(text) > API_LIMITS["文字長度限制"]:
            debug_print("文字過長，截斷處理")
            text = text[:API_LIMITS["文字長度限制"]]
        
        api_config = ITHUAN_API["標音服務"]
        url = f"{api_config['網域']}{api_config['端點']}"
        
        data = {'taibun': text.strip()}
        
        debug_print(f"API 請求: {url}")
        
        api_start = time.time()
        response = requests.post(
            url,
            data=data,
            headers={
                'Content-Type': api_config['內容類型'],
                'User-Agent': 'TaiwaneseVoiceChat/1.0'
            },
            timeout=15
        )
        api_time = time.time() - api_start
        log_step_time("　├─ 意傳標音API", api_time, f"狀態: {response.status_code}")
        
        debug_print(f"回應狀態: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            if 'kiatko' in result and result['kiatko']:
                romanization_parts = []
                for item in result['kiatko']:
                    if 'KIP' in item and item['KIP']:
                        romanization_parts.append(item['KIP'])
                
                if romanization_parts:
                    romanization = ' '.join(romanization_parts)
                    debug_print(f"羅馬拼音: {romanization}")
                    return romanization, result.get('分詞', text), result['kiatko']
            
            if '分詞' in result:
                segmented = result['分詞']
                debug_print(f"分詞結果: {segmented}")
                return segmented, segmented, []
        
        debug_print("API 返回異常")
        return text, text, []
        
    except Exception as e:
        debug_print(f"標音 API 失敗: {e}")
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



@performance_timer("LLM智能對話")
def chat_with_ollama(text):
    """與 Ollama LLM 對話"""
    try:
        debug_print(f"LLM 對話處理: '{text}'")
        
        # 使用繁體中文回應，限制15字以內
        prompt = f"""你是一個台灣人工智慧助理，請用繁體中文回應。注意：
1. 使用自然、流暢的繁體中文
2. 不要使用簡體字
3. 回答必須在15個字以內
4. 保持對話的連貫性和邏輯性

用戶：{text}

助理："""

        api_start = time.time()
        
        # 使用性能配置的優化參數
        if PERFORMANCE_CONFIG_AVAILABLE:
            config = get_current_config()
            llm_options = config["llm_config"].copy()
            timeout = llm_options.pop("timeout", 15)
            debug_print(f"使用性能配置LLM參數: {llm_options}")
        else:
            # 備用優化參數
            llm_options = {
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 40,
                'repeat_penalty': 1.1,
                'num_thread': 4,
                'num_batch': 512,
            }
            timeout = 15
        
        response = requests.post('http://localhost:11434/api/generate', 
            json={
                'model': 'gemma3:4b',
                'prompt': prompt,
                'stream': False,
                'options': llm_options
            },
            timeout=timeout
        )
        api_time = time.time() - api_start
        log_step_time("　├─ Ollama API請求", api_time)
        
        if response.status_code == 200:
            result = response.json()
            reply = result.get('response', '').strip()
            
            # 清理回應，但保留標點符號
            cleaned_reply = re.sub(r'[^\u4e00-\u9fff！？。，、：；「」『』（）]', '', reply)
            
            # 如果清理後的回應為空，返回預設回應
            if not cleaned_reply:
                final_reply = "好的！"
            else:
                # 限制回應在15個字以內
                if len(cleaned_reply) > 15:
                    final_reply = cleaned_reply[:15]
                else:
                    final_reply = cleaned_reply
                
            debug_print(f"LLM 回應: '{final_reply}'")
            return final_reply
        else:
            debug_print(f"LLM API 失敗: {response.status_code}")
            return "好的！"
            
    except Exception as e:
        debug_print(f"LLM 對話失敗: {e}")
        return "好的！"

@performance_timer("意傳科技TTS服務")
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
    """處理語音檔案"""
    global remote_tts_service, romanization_converter
    
    # 總體計時開始
    total_start_time = time.time()
    print(f"🚀 開始處理語音請求 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 各步驟計時統計
    step_times = {}
    
    try:
        # 步驟0: 請求驗證
        step_start = time.time()
        if 'audio' not in request.files:
            return jsonify({'error': '沒有收到音檔'}), 400
            
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'error': '音檔名稱為空'}), 400
        step_times['請求驗證'] = time.time() - step_start
        log_step_time("請求驗證", step_times['請求驗證'])
            
        # 步驟1: 音檔保存
        step_start = time.time()
        # 建立本地保存目錄
        os.makedirs("uploads", exist_ok=True)
        
        # 決定檔案副檔名
        content_type = audio_file.content_type or 'audio/webm'
        if 'webm' in content_type:
            suffix = '.webm'
        elif 'wav' in content_type:
            suffix = '.wav'
        elif 'mp3' in content_type:
            suffix = '.mp3'
        else:
            suffix = '.audio'
        
        # 保存音檔
        timestamp = int(time.time() * 1000)
        local_filename = f"uploads/recording_{timestamp}{suffix}"
        
        audio_file.seek(0)
        audio_data = audio_file.read()
        
        if len(audio_data) == 0:
            return jsonify({'error': '音檔數據為空'}), 400
        
        try:
            with open(local_filename, 'wb') as f:
                f.write(audio_data)
            debug_print(f"音檔保存成功: {local_filename}")
        except Exception as e:
            debug_print(f"音檔保存失敗: {e}")
            return jsonify({'error': '音檔保存失敗'}), 500
        
        if not os.path.exists(local_filename) or os.path.getsize(local_filename) == 0:
            return jsonify({'error': '保存的音檔無效'}), 400
        
        step_times['音檔保存'] = time.time() - step_start
        log_step_time("音檔保存", step_times['音檔保存'], f"檔案大小: {len(audio_data)} bytes")
        
        try:
            debug_print("開始台語語音對話處理")
            
            # 步驟2: 音檔格式轉換（如果需要）
            step_start = time.time()
            audio_path = local_filename
            if suffix != '.wav':
                converted_path = convert_webm_with_ffmpeg(local_filename)
                if converted_path:
                    audio_path = converted_path
                else:
                    return jsonify({'error': '音檔格式轉換失敗'}), 400
            step_times['格式轉換'] = time.time() - step_start
            log_step_time("音檔格式轉換", step_times['格式轉換'])
            
            # 步驟3: 台語語音辨識
            step_start = time.time()
            recognized_text = transcribe_taiwanese_audio(audio_path)
            step_times['語音辨識'] = time.time() - step_start
            log_step_time("台語語音辨識", step_times['語音辨識'], f"辨識結果: '{recognized_text}'")
            
            if not recognized_text:
                return jsonify({'error': '無法辨識台語語音內容'}), 400
            
            # 步驟4: LLM 對話
            step_start = time.time()
            ai_response = chat_with_ollama(recognized_text)
            step_times['LLM對話'] = time.time() - step_start
            log_step_time("LLM智能對話", step_times['LLM對話'], f"AI回應: '{ai_response}'")
            
            # 步驟5: 台語標音轉換
            step_start = time.time()
            romanization, segmented, kiatko_data = get_taiwanese_pronunciation(ai_response)
            step_times['標音轉換'] = time.time() - step_start
            log_step_time("台語標音轉換", step_times['標音轉換'], f"羅馬拼音: '{romanization}'")
            
            # 步驟6: 格式轉換（羅馬拼音轉數字調）
            step_start = time.time()
            if romanization_converter:
                numeric_tone_text = romanization_converter.convert_to_numeric_tone(romanization)
                debug_print(f"格式轉換: '{romanization}' -> '{numeric_tone_text}'")
            else:
                numeric_tone_text = romanization
                debug_print(f"跳過格式轉換: '{romanization}'")
            step_times['格式轉換'] = time.time() - step_start
            log_step_time("羅馬拼音格式轉換", step_times['格式轉換'], f"數字調格式: '{numeric_tone_text}'")
            
            # 步驟7: 文字轉語音（使用自訓練遠端 TTS 服務）
            step_start = time.time()
            print(f"\n🔊 步驟7: 台語語音合成")
            print(f"使用自訓練遠端 TTS 服務 (163.13.202.125:5000)")
            audio_file_path = None
            if remote_tts_service:
                audio_file_path = remote_tts_service.generate_speech(numeric_tone_text)
            else:
                print("⚠️ 遠端TTS服務未初始化，使用意傳科技TTS作為備用")
                audio_file_path = text_to_speech_ithuan(romanization, kiatko_data)
            step_times['語音合成'] = time.time() - step_start
            log_step_time("台語語音合成", step_times['語音合成'], f"音檔: {audio_file_path if audio_file_path else '失敗'}")
            
            if audio_file_path:
                print(f"🔊 TTS 成功: {audio_file_path}")
            else:
                print("⚠️ TTS 失敗")
            
            # 計算總耗時
            total_time = time.time() - total_start_time
            
            # 步驟8: 返回結果
            print(f"\n✅ 台語語音對話處理完成")
            print(f"🎯 總處理時間: {total_time:.3f}秒")
            print("📊 各步驟耗時統計:")
            for step_name, duration in step_times.items():
                percentage = (duration / total_time) * 100
                print(f"   • {step_name}: {duration:.3f}秒 ({percentage:.1f}%)")
            
            # 生成性能優化建議
            optimization_suggestions = []
            if PERFORMANCE_CONFIG_AVAILABLE:
                optimization_suggestions = get_optimization_suggestions(step_times, total_time)
            
            result = {
                'success': True,
                'transcription': recognized_text,
                'ai_response': ai_response,
                'romanization': romanization,
                'numeric_tone_text': numeric_tone_text,
                'segmented': segmented,
                'kiatko_count': len(kiatko_data),
                'audio_url': f'/{audio_file_path}' if audio_file_path else None,
                'api_info': f"使用自訓練遠端 TTS 服務 (163.13.202.125:5000)" if remote_tts_service and audio_file_path else "使用意傳科技TTS作為備用",
                'performance_stats': {
                    'total_time': total_time,
                    'step_times': step_times,
                    'bottleneck': max(step_times, key=step_times.get) if step_times else None,
                    'mode': PERFORMANCE_MODE,
                    'suggestions': optimization_suggestions
                }
            }
            
            debug_print("台語語音對話處理完成")
            return jsonify(result)
            
        finally:
            # 清理臨時檔案
            try:
                if CLEANUP_FILES:
                    if os.path.exists(local_filename):
                        os.unlink(local_filename)
                        debug_print(f"清理本地檔案: {local_filename}")
                    if audio_path != local_filename and os.path.exists(audio_path):
                        os.unlink(audio_path)
                        debug_print(f"清理轉換檔案: {audio_path}")
            except Exception as e:
                debug_print(f"清理檔案失敗: {e}")
        
    except Exception as e:
        debug_print(f"處理錯誤: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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
            'api_status': 'success' if audio_file else 'failed',
            'api_info': {
                '標音服務': f"{ITHUAN_API['標音服務']['網域']}{ITHUAN_API['標音服務']['端點']}",
                '遠端TTS': "未配置" # config.get_remote_tts_display_name() if remote_tts_service and config.is_remote_tts_configured() else "未配置"
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e), 'api_status': 'failed'})

@app.route('/test_remote_tts')
def test_remote_tts():
    """測試遠端TTS服務的不同參數"""
    global remote_tts_service
    
    if not remote_tts_service:
        return jsonify({'error': '遠端TTS服務未初始化'}), 500
    
    try:
        test_text = request.args.get('text', 'li2 ho2')
        
        # 解析額外參數
        additional_params = {}
        for key, value in request.args.items():
            if key != 'text':
                additional_params[key] = value
        
        print(f"🧪 測試遠端TTS: '{test_text}'")
        if additional_params:
            print(f"📝 額外參數: {additional_params}")
        
        # 使用測試方法
        audio_file = remote_tts_service.test_with_params(test_text, additional_params)
        
        result = {
            'test_text': test_text,
            'additional_params': additional_params,
            'audio_file': audio_file,
            'api_status': 'success' if audio_file else 'failed',
            'remote_url': f"{remote_tts_service.base_url}{remote_tts_service.endpoint}",
            'usage': {
                'basic': '/test_remote_tts?text=li2%20ho2',
                'with_params': '/test_remote_tts?text=li2%20ho2&version=1&model=default'
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
    print("API 配置資訊:")
    for service_name, config in ITHUAN_API.items():
        print(f"   {service_name}: {config['網域']}{config['端點']} ({config['方法']})")
    
    print(f"⚠️ 使用限制: 每IP每分鐘最多{API_LIMITS['每分鐘下載限制']}次音檔下載")
    print("=" * 50)
    
    # 初始化台語模型
    model_ready = init_taiwanese_model()
    
    # 初始化TTS服務和格式轉換器
    
    print("初始化自訓練遠端TTS服務...")
    try:
        remote_tts_service = RemoteTtsService()
        print("遠端TTS服務初始化成功")
        
        # 測試連線
        if remote_tts_service.test_connection():
            print("遠端TTS服務連線測試成功")
        else:
            print("遠端TTS服務連線測試失敗")
            
    except Exception as e:
        print(f"遠端TTS服務初始化失敗: {e}")
        print("系統無法正常運作")
        remote_tts_service = None
    
    print("初始化羅馬拼音格式轉換器...")
    try:
        romanization_converter = RomanizationConverter()
        print("羅馬拼音轉換器初始化成功")
    except Exception as e:
        print(f"羅馬拼音轉換器初始化失敗: {e}")
        print("將直接使用原始格式")
        romanization_converter = None
    
    if model_ready:
        print("系統初始化完成！")
        print("啟動 Web 服務...")
        print("訪問 http://localhost:5000 開始使用")
        print("訪問 http://localhost:5000/test_api 測試 API")
        print("=" * 50)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("系統初始化失敗，無法啟動 Web 服務")
        print("請檢查模型安裝和相關依賴") 