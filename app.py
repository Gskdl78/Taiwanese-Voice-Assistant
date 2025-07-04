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
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import torch
import requests
from urllib.parse import urlencode, quote

app = Flask(__name__)

# 全域變數
taiwanese_processor = None
taiwanese_model = None
device = None
ffmpeg_path = None
pipeline_asr = None

# 設定常數
CLEANUP_FILES = False  # 是否清理暫存檔案
DEBUG_MODE = True      # 是否顯示詳細調試信息

# 意傳科技 API 設定
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

# API使用限制
API_LIMITS = {
    "每分鐘下載限制": 3,
    "文字長度限制": 200,
    "避免同時請求": True
}

def debug_print(message):
    """條件式調試輸出"""
    if DEBUG_MODE:
        print(message)

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
    global taiwanese_processor, taiwanese_model, device, ffmpeg_path, pipeline_asr
    
    debug_print("初始化台語語音辨識系統...")
    
    # 檢查設備
    device = "cuda" if torch.cuda.is_available() else "cpu"
    debug_print(f"設備: {device}")
    
    # 檢查 FFmpeg
    ffmpeg_path = find_ffmpeg()
    debug_print(f"FFmpeg: {ffmpeg_path if ffmpeg_path else '未找到'}")
    
    # 台語專門模型清單
    models_to_try = [
        "NUTN-KWS/Whisper-Taiwanese-model-v0.5",
        "EricChang/TAT-TD-openai-whisper-large-v2-Lora-epoch5-total5epoch",
        "cathyi/tw-tw-openai-whisper-large-v2-Lora-epoch5-total5epoch",
    ]
    
    debug_print("載入台語專門模型...")
    
    # 嘗試載入台語模型
    for model_name in models_to_try:
        try:
            debug_print(f"嘗試載入: {model_name}")
            
            # 方法1: 使用 transformers 直接載入
            try:
                processor = WhisperProcessor.from_pretrained(model_name)
                model = WhisperForConditionalGeneration.from_pretrained(model_name)
                model = model.to(device)
                
                taiwanese_processor = processor
                taiwanese_model = model
                debug_print(f"成功載入台語模型: {model_name}")
                return True
                
            except Exception:
                # 方法2: 使用 pipeline
                try:
                    pipeline_asr = pipeline(
                        "automatic-speech-recognition",
                        model=model_name,
                        device=0 if device == "cuda" else -1,
                        return_timestamps=True
                    )
                    taiwanese_processor = "pipeline"
                    debug_print(f"成功載入台語模型 (pipeline): {model_name}")
                    return True
                except Exception:
                    continue
                    
        except Exception as e:
            debug_print(f"載入 {model_name} 失敗: {e}")
            continue
    
    # 使用標準 Whisper
    debug_print("台語模型載入失敗，使用標準 Whisper")
    try:
        import whisper
        taiwanese_model = whisper.load_model("base")
        taiwanese_processor = "whisper_direct"
        debug_print("標準 Whisper 模型載入成功")
        return True
        
    except Exception as e:
        debug_print(f"所有模型載入失敗: {e}")
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
        debug_print("使用 FFmpeg 轉換...")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_wav_path = temp_wav.name
        
        cmd = [
            ffmpeg_path, '-i', webm_file, 
            '-ar', '16000',
            '-ac', '1',
            '-y',
            temp_wav_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            debug_print("FFmpeg 轉換成功")
            audio, sr = librosa.load(temp_wav_path, sr=16000)
            os.unlink(temp_wav_path)
            return audio, sr
        else:
            debug_print(f"FFmpeg 轉換失敗: {result.stderr}")
            if os.path.exists(temp_wav_path):
                os.unlink(temp_wav_path)
            return None
            
    except Exception as e:
        debug_print(f"FFmpeg 轉換出錯: {e}")
        return None

def transcribe_taiwanese_audio(audio_file_path):
    """台語語音辨識"""
    global taiwanese_processor, taiwanese_model, pipeline_asr, device
    
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
            result = convert_webm_with_ffmpeg(audio_file_path)
            if result:
                audio_data, sr = result
            else:
                debug_print("WebM 轉換失敗")
                return ""
        else:
            try:
                audio_data, sr = librosa.load(audio_file_path, sr=16000)
                debug_print("音檔載入成功")
            except Exception as e:
                debug_print(f"音檔載入失敗: {e}")
                return ""
        
        if audio_data is None or len(audio_data) == 0:
            debug_print("音檔資料為空")
            return ""
        
        debug_print(f"音檔資訊: 長度={len(audio_data)}, 取樣率={sr}")
        
        transcription = ""
        
        # 方法1: 使用台語 transformers 模型
        if taiwanese_processor and taiwanese_model and taiwanese_processor != "pipeline" and taiwanese_processor != "whisper_direct":
            try:
                debug_print("使用台語 transformers 模型...")
                inputs = taiwanese_processor(audio_data, sampling_rate=sr, return_tensors="pt")
                inputs = inputs.to(device)
                
                with torch.no_grad():
                    predicted_ids = taiwanese_model.generate(inputs["input_features"])
                    transcription = taiwanese_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
                debug_print(f"transformers 辨識成功: '{transcription}'")
                
            except Exception as e:
                debug_print(f"transformers 模型失敗: {e}")
        
        # 方法2: 使用 pipeline
        if not transcription and pipeline_asr and taiwanese_processor == "pipeline":
            try:
                debug_print("使用 pipeline 模型...")
                temp_wav = f"temp_{int(time.time())}.wav"
                sf.write(temp_wav, audio_data, sr)
                
                result = pipeline_asr(temp_wav)
                transcription = result["text"]
                
                if os.path.exists(temp_wav):
                    os.unlink(temp_wav)
                
                debug_print(f"pipeline 辨識成功: '{transcription}'")
                
            except Exception as e:
                debug_print(f"pipeline 模型失敗: {e}")
        
        # 方法3: 使用標準 Whisper
        if not transcription and taiwanese_model and taiwanese_processor == "whisper_direct":
            try:
                debug_print("使用標準 Whisper 模型...")
                temp_wav = f"temp_{int(time.time())}.wav"
                sf.write(temp_wav, audio_data, sr)
                
                result = taiwanese_model.transcribe(temp_wav, language="zh")
                transcription = result["text"]
                
                if os.path.exists(temp_wav):
                    os.unlink(temp_wav)
                
                debug_print(f"Whisper 辨識成功: '{transcription}'")
                
            except Exception as e:
                debug_print(f"Whisper 模型失敗: {e}")
        
        if transcription:
            transcription = transcription.strip()
            debug_print(f"最終辨識結果: '{transcription}'")
            return transcription
        else:
            debug_print("所有辨識方法都失敗")
            return ""
            
    except Exception as e:
        debug_print(f"語音辨識出錯: {e}")
        return ""

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
        
        response = requests.post(
            url,
            data=data,
            headers={
                'Content-Type': api_config['內容類型'],
                'User-Agent': 'TaiwaneseVoiceChat/1.0'
            },
            timeout=15
        )
        
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

def check_audio_response(response):
    """檢查響應是否包含音檔數據"""
    if len(response.content) <= 100:
        return False, None
    
    content_type = response.headers.get('content-type', '').lower()
    
    # 檢查是否為音訊內容
    is_audio = (
        'audio' in content_type or 
        len(response.content) > 1000 or
        response.content.startswith(b'RIFF') or  # WAV檔案標頭
        response.content.startswith(b'ID3') or   # MP3檔案標頭
        response.content.startswith(b'\xff\xfb') # MP3檔案標頭變種
    )
    
    return is_audio, content_type

def save_audio_file(content, filename_prefix, additional_info=""):
    """保存音檔並返回檔案路徑"""
    try:
        timestamp = int(time.time())
        safe_filename = re.sub(r'[<>:"/\\|?*,]', '_', additional_info.replace(' ', '_').replace('-', '_'))[:20]
        output_file = f"static/{filename_prefix}_{safe_filename}_{timestamp}.wav"
        
        os.makedirs("static", exist_ok=True)
        
        with open(output_file, 'wb') as f:
            f.write(content)
        
        debug_print(f"音檔保存成功: {output_file}")
        return output_file
        
    except Exception as e:
        debug_print(f"音檔保存失敗: {e}")
        return None

def make_ithuan_tts_request(api_config_key, params):
    """統一的意傳科技 TTS 請求函數"""
    try:
        api_config = ITHUAN_API[api_config_key]
        base_url = f"{api_config['網域']}{api_config['端點']}"
        
        if api_config_key == "整段語音合成":
            encoded_params = urlencode(params, safe='', encoding='utf-8')
            full_url = f"{base_url}?{encoded_params}"
        else:  # 單詞語音合成
            encoded_taibun = quote(params['taibun'], safe='')
            full_url = f"{base_url}?taibun={encoded_taibun}"
        
        debug_print(f"TTS API: {full_url}")
        
        response = requests.get(
            full_url,
            timeout=30,
            headers={
                'User-Agent': 'TaiwaneseVoiceChat/1.0',
                'Accept': 'audio/wav, audio/*, */*',
                'Accept-Language': 'zh-TW,zh;q=0.9,en;q=0.8'
            }
        )
        
        debug_print(f"回應狀態: {response.status_code}, 大小: {len(response.content)} bytes")
        
        is_audio, content_type = check_audio_response(response)
        
        if is_audio:
            prefix = "tts_full" if api_config_key == "整段語音合成" else "tts_word"
            additional_info = params.get('查詢語句', '') or params.get('taibun', '')
            
            audio_file = save_audio_file(response.content, prefix, additional_info)
            if audio_file:
                debug_print(f"{api_config_key}成功")
                return audio_file
        else:
            debug_print(f"{api_config_key}回應不是音檔格式")
            if response.content:
                try:
                    text_content = response.content.decode('utf-8', errors='ignore')[:200]
                    debug_print(f"回應內容: {text_content}")
                except:
                    pass
        
        return None
        
    except Exception as e:
        debug_print(f"{api_config_key}失敗: {e}")
        return None

def text_to_speech_ithuan_full_sentence(腔口, 分詞):
    """整段語音合成"""
    params = {
        '查詢腔口': 腔口,
        '查詢語句': 分詞
    }
    return make_ithuan_tts_request("整段語音合成", params)

def text_to_speech_ithuan_single_word(羅馬拼音):
    """單詞語音合成"""
    params = {'taibun': 羅馬拼音}
    return make_ithuan_tts_request("單詞語音合成", params)

def chat_with_ollama(text):
    """與 Ollama LLM 對話"""
    try:
        debug_print(f"LLM 對話處理: '{text}'")
        
        prompt = f"""你是一個親切的台語助手。請用台語漢字簡短回應以下話語：

規則：
- 只能回應 3-8 個字
- 使用常見的繁體中文漢字
- 語氣要親切自然
- 不要重複同一個字超過 2 次
- 要符合台語的說話習慣

用戶說：{text}

台語回應："""

        response = requests.post('http://localhost:11434/api/generate', 
            json={
                'model': 'gemma3:4b',
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.6,
                    'num_predict': 30,
                    'top_p': 0.9
                }
            },
            timeout=20
        )
        
        if response.status_code == 200:
            result = response.json()
            reply = result.get('response', '').strip()
            
            # 清理回應
            cleaned_reply = re.sub(r'[^\u4e00-\u9fff！？。，、]', '', reply)
            
            # 移除重複超過 2 次的字符
            def remove_excessive_repeats(text):
                result = ""
                char_count = {}
                for char in text:
                    char_count[char] = char_count.get(char, 0) + 1
                    if char_count[char] <= 2:
                        result += char
                return result
            
            cleaned_reply = remove_excessive_repeats(cleaned_reply)
            
            if len(cleaned_reply) > 8:
                cleaned_reply = cleaned_reply[:8]
            
            if len(cleaned_reply) < 2:
                final_reply = "好欸！"
            else:
                final_reply = cleaned_reply
                
            debug_print(f"LLM 回應: '{final_reply}'")
            return final_reply
        else:
            debug_print(f"LLM API 失敗: {response.status_code}")
            return "好欸！"
            
    except Exception as e:
        debug_print(f"LLM 對話失敗: {e}")
        return "好欸！"

def text_to_speech_ithuan(text, kiatko_data=None):
    """意傳科技 TTS 主函數"""
    debug_print(f"意傳科技 TTS 開始: '{text}'")
    
    # 優先嘗試整段語音合成
    full_audio = text_to_speech_ithuan_full_sentence("閩南語", text)
    if full_audio:
        debug_print("整段語音合成成功")
        return full_audio
    
    # 嘗試單詞語音合成
    if kiatko_data and len(kiatko_data) > 0:
        first_word = kiatko_data[0]
        if 'KIP' in first_word and first_word['KIP']:
            single_audio = text_to_speech_ithuan_single_word(first_word['KIP'])
            if single_audio:
                debug_print("單詞語音合成成功")
                return single_audio
    
    debug_print("所有 TTS 方案都失敗")
    return None

@app.route('/')
def index():
    """主頁面"""
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """處理語音檔案"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': '沒有收到音檔'}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'error': '音檔名稱為空'}), 400
            
        # 建立本地保存目錄
        os.makedirs("uploads", exist_ok=True)
        
        # 決定檔案副檔名
        content_type = getattr(audio_file, 'content_type', 'unknown')
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
        
        try:
            debug_print("開始台語語音對話處理")
            
            # 1. 台語語音辨識
            recognized_text = transcribe_taiwanese_audio(local_filename)
            if not recognized_text:
                return jsonify({'error': '無法辨識台語語音內容'}), 400
            
            # 2. LLM 對話
            ai_response = chat_with_ollama(recognized_text)
            
            # 3. 台語標音轉換
            romanization, segmented, kiatko_data = get_taiwanese_pronunciation(ai_response)
            
            # 4. 文字轉語音
            audio_file_path = text_to_speech_ithuan(romanization, kiatko_data)
            
            # 5. 返回結果
            result = {
                'recognized_text': recognized_text,
                'ai_response': ai_response,
                'romanization': romanization,
                'segmented': segmented,
                'kiatko_count': len(kiatko_data),
                'audio_url': f'/{audio_file_path}' if audio_file_path else None,
                'api_info': f"使用意傳科技 API (限制: {API_LIMITS['每分鐘下載限制']}次/分鐘)"
            }
            
            debug_print("台語語音對話處理完成")
            return jsonify(result)
            
        finally:
            # 清理本地檔案
            if CLEANUP_FILES and os.path.exists(local_filename):
                try:
                    os.unlink(local_filename)
                    debug_print(f"清理本地檔案: {local_filename}")
                except Exception as e:
                    debug_print(f"清理檔案失敗: {e}")
        
    except Exception as e:
        debug_print(f"處理錯誤: {e}")
        return jsonify({'error': f'處理失敗: {str(e)}'}), 500

@app.route('/test_api')
def test_api():
    """測試意傳科技 API"""
    try:
        test_text = "你好嗎"
        debug_print(f"測試意傳科技 API: '{test_text}'")
        
        # 測試標音 API
        romanization, segmented, kiatko_data = get_taiwanese_pronunciation(test_text)
        
        # 測試 TTS API
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
    print("啟動台語語音對話 Web 應用程式")
    print("整合意傳科技 API（基於 TauPhahJi-BangTsam 規範）")
    print("=" * 50)
    
    # 顯示API配置資訊
    print("API 配置資訊:")
    for service_name, config in ITHUAN_API.items():
        print(f"   {service_name}: {config['網域']}{config['端點']} ({config['方法']})")
    
    print(f"使用限制: 每IP每分鐘最多{API_LIMITS['每分鐘下載限制']}次音檔下載")
    print("=" * 50)
    
    # 初始化台語模型
    model_ready = init_taiwanese_model()
    
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