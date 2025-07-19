# 台語語音對話系統

專門為台語語音辨識與智能對話打造的 Web 應用系統，整合多項先進技術提供流暢的台語語音互動體驗。

## 核心特色

- **專業台語支援**：使用專門訓練的台語語音模型，真正理解台語語音特色
- **智能對話**：整合大型語言模型，提供自然流暢的對話體驗  
- **自訓練TTS引擎**：使用自訓練的遠端 TTS 服務，確保語音輸出品質
- **現代化介面**：響應式 Web 界面，支援桌面與行動裝置
- **GPU 加速**：支援 CUDA GPU 加速，提供更快的語音辨識體驗

## 系統架構

### 完整處理流程
```
台語語音輸入 → STT辨識 → LLM智能對話 → 台語標音 → 格式轉換 → TTS語音合成 → 台語語音輸出
     ↓           ↓          ↓           ↓         ↓          ↓           ↓
  .webm錄音   中文文字   智能中文回應  台羅拼音   數字調格式  .wav音檔   台語語音播放
```

### 技術堆疊
- **前端**：HTML5 + JavaScript + Web Audio API
- **後端**：Flask + Python 3.8+
- **STT引擎**：Whisper (台語專門模型 NUTN-KWS/Whisper-Taiwanese-model-v0.5)
- **LLM對話**：Ollama + Gemma/Llama3系列
- **標音服務**：意傳科技標音 API
- **TTS引擎**：自訓練遠端 TTS 服務
- **格式轉換**：自研羅馬拼音轉換器

## 專案結構

```
台語語音對話系統/ 
├── app.py                      # 主應用程式 (Flask Web服務)
├── remote_tts_service.py       # 遠端TTS服務模組
├── romanization_converter.py   # 羅馬拼音格式轉換器
├── requirements.txt            # Python依賴套件清單
├── README.md                  # 專案說明文件
├── LICENSE                    # 開源授權條款
│
├── templates/                 # 網頁模板目錄
│   └── index.html            # 主要語音對話界面
│
├── static/                   # 動態生成檔案目錄
│   └── (TTS生成的音檔)        # 系統自動生成的語音檔案
│
├── uploads/                  # 上傳檔案目錄  
│   └── (用戶錄音檔案)         # 用戶錄製的語音檔案
│
├── ffmpeg_bin/               # 音訊處理工具
│   └── ffmpeg.exe           # FFmpeg二進位檔案
│
└── TauPhahJi-API-docs/       # 意傳科技API文檔
    ├── API和組件文檔.md       # API規範文檔
    ├── 專案說明書.md          # 專案說明
    └── 狀態管理和部署指南.md   # 部署指南
```

## 安裝與設定

### 前置需求

1. **Python 3.8+** 環境
2. **CUDA 支援的 GPU**（建議）
3. **網路連線** (存取意傳科技 API 和遠端 TTS)
4. **麥克風權限** (瀏覽器語音錄製)
5. **Ollama 服務** (LLM 對話引擎)

### 安裝步驟

1. 安裝 Python 依賴：
```bash
pip install -r requirements.txt
```

2. 設定 Ollama LLM 服務：
```bash
# 安裝 Ollama (Windows)
# 下載並安裝: https://ollama.ai/download/windows

# 啟動 Ollama 服務
ollama serve

# 下載模型
ollama pull gemma3:4b
```

3. 確保 FFmpeg 在 `ffmpeg_bin/` 目錄中

4. 啟動應用程式：
```bash
python app.py
```

5. 訪問：http://localhost:5000

## 效能指標

典型處理時間分布：
- 語音辨識：~0.4-0.6 秒 (GPU)
- LLM 對話：~1.0-1.5 秒
- 標音轉換：~0.6-0.7 秒
- 語音合成：~0.5-1.5 秒
- 總處理時間：~2.5-4.0 秒

## 授權與致謝

### 開源授權
本專案採用 MIT License 授權條款。

### 致謝
- **意傳科技 (iTaigi)**：提供台語標音 API
- **國立台南大學 (NUTN)**：台語專門 Whisper 模型
- **Ollama 團隊**：本地 LLM 推理引擎
- **OpenAI Whisper**：語音辨識基礎模型
