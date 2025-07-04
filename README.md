# 台語語音對話系統

專門為台語語音辨識與智能對話打造的**超輕量化**Web應用系統，整合多項先進技術提供流暢的台語語音互動體驗。

## 核心特色

- **專業台語支援**：使用專門訓練的台語語音模型，真正理解台語語音特色
- **智能對話**：整合大型語言模型，提供自然流暢的對話體驗  
- **高品質TTS**：意傳科技API + 多重備用方案，確保語音輸出穩定性
- **現代化介面**：響應式Web界面，支援桌面與行動裝置
- **即時處理**：毫秒級語音辨識與合成，接近真實對話體驗
- **超輕量化**：專案體積精簡99.9%，啟動更快、部署更簡便

## 系統架構

### 完整處理流程
```
台語語音輸入 → STT辨識 → LLM智能對話 → 台語標音 → TTS語音合成 → 台語語音輸出
     ↓           ↓          ↓           ↓          ↓           ↓
  .webm錄音   中文文字   智能中文回應  台語羅馬拼音  .wav音檔   台語語音播放
```

### 技術堆疊
- **前端**：HTML5 + JavaScript + Web Audio API
- **後端**：Flask + Python 3.8+
- **STT引擎**：Whisper (台語專門模型)
- **LLM對話**：Ollama + Gemma/Llama3系列
- **標音服務**：意傳科技標音API
- **TTS引擎**：意傳科技TTS API + Windows SAPI + Google TTS

## 專案結構 (超精簡版)

```
台語語音對話系統/ 
├── app.py                          # 主應用程式 (Flask Web服務)
├── requirements.txt                # Python依賴套件清單
├── README.md                      # 專案說明文件
├── LICENSE                        # 開源授權條款
├── .gitignore                     # Git版本控制忽略檔案
│
├── templates/                     # 網頁模板目錄
│   └── index.html                # 主要語音對話界面
│
├── static/                       # 動態生成檔案目錄
│   └── (TTS生成的音檔)            # 系統自動生成的語音檔案
│
├── uploads/                      # 上傳檔案目錄  
│   └── (用戶錄音檔案)             # 用戶錄製的語音檔案
│
├── ffmpeg_bin/                   # 音訊處理工具 (需要自行下載)
│   └── ffmpeg.exe               # FFmpeg二進位檔案 (用戶自行放置)
│
└── TauPhahJi-API-docs/           # 意傳科技API文檔 (僅79.6KB)
    ├── API和組件文檔.md          # API規範文檔
    ├── 專案說明書.md             # 專案說明
    ├── 專案完整說明書.md         # 完整技術文檔
    ├── 狀態管理和部署指南.md     # 部署指南
    ├── README.md                # 專案介紹
    └── LICENSE                  # 授權條款
```

## 專案優勢

### **超輕量化設計**
- **99.9%體積縮減**：從~104MB縮減至<1MB核心檔案
- **極速啟動**：系統啟動時間大幅縮短
- **簡化部署**：最少依賴，部署更容易
- **維護友善**：精簡結構，問題定位更快

### **專業台語技術**
- **專門模型**：台語專用Whisper-v0.5，辨識精度顯著優於通用模型
- **API整合**：意傳科技專業台語標音與TTS服務
- **404修復**：已解決TTS API錯誤音檔捕獲問題
- **多重備援**：三層TTS備用機制確保穩定性

## 快速開始

### 前置需求

1. **Python 3.8+** 環境
2. **網路連線** (存取意傳科技API)
3. **麥克風權限** (瀏覽器語音錄製)
4. **Ollama服務** (LLM對話引擎)

### 安裝步驟

#### 1. 安裝Python依賴

```bash
pip install -r requirements.txt
```

主要套件包含：
- `flask` - Web框架
- `requests` - HTTP客戶端
- `transformers` - Hugging Face模型庫
- `torch` - PyTorch深度學習框架
- `librosa` - 音訊處理庫

#### 2. 設定Ollama LLM服務

```bash
# 安裝Ollama (Windows)
# 下載並安裝: https://ollama.ai/download/windows

# 啟動Ollama服務
ollama serve

# 下載推薦的LLM模型 (選擇其一)
ollama pull gemma2:2b        # 輕量級模型 (推薦)
ollama pull llama3.2:3b      # 平衡性能模型
ollama pull gemma3:4b        # 高性能模型
```

#### 3. 下載FFmpeg (音訊處理工具)

```bash
# 下載FFmpeg並放置到專案目錄
# 1. 訪問 https://ffmpeg.org/download.html
# 2. 下載Windows版本
# 3. 解壓縮後將ffmpeg.exe放到 ffmpeg_bin/ 目錄中
# 4. 或使用Chocolatey: choco install ffmpeg
```

#### 4. 啟動應用程式

```bash
python app.py
```

系統會自動：
- 啟動Flask Web服務器 (端口5000)
- 連接Ollama LLM服務
- 檢查所有API連線狀態
- 準備語音對話界面

#### 5. 開始使用

瀏覽器訪問：**http://localhost:5000**

## 詳細使用指南

### 語音對話操作

1. **錄製台語語音**
   - 點擊紅色「開始錄音」按鈕
   - 清楚說出台語內容 (建議3-10秒)
   - 點擊「停止錄音」完成錄製

2. **語音辨識處理**  
   - 系統自動上傳錄音檔案
   - 使用台語專門Whisper模型進行STT
   - 將台語語音轉換為中文文字

3. **智能對話生成**
   - 發送中文文字到Ollama LLM
   - AI理解語意並生成適當回應
   - 支援多輪對話與上下文理解

4. **台語標音轉換**
   - 呼叫意傳科技標音API
   - 將中文回應轉換為台語羅馬拼音
   - 確保正確的台語發音標示

5. **台語語音合成**
   - 使用台語羅馬拼音進行TTS
   - 多重引擎確保合成成功率
   - 自動播放生成的台語語音

## 技術規格詳解

### STT (語音轉文字)

**模型資訊**：
- **名稱**：NUTN-KWS/Whisper-Taiwanese-model-v0.5
- **類型**：台語專門微調的Whisper模型  
- **訓練數據**：台語語音資料集
- **支援格式**：WebM, WAV, MP3
- **辨識精度**：針對台語優化，顯著優於通用模型

**處理流程**：
```python
音檔上傳 → FFmpeg轉換 → Whisper推理 → 中文文字輸出
```

### LLM對話引擎

**支援模型**：
- **Gemma2:2b** - 輕量快速，適合即時對話
- **Llama3.2:3b** - 平衡性能與速度
- **Gemma3:4b** - 高品質回應，較耗資源

**對話特色**：
- 上下文記憶 (多輪對話)
- 台語文化理解
- 自然語言生成
- 即時回應 (< 2秒)

### 標音服務

**API提供者**：意傳科技 (iTaigi)
- **端點**：https://hokbu.ithuan.tw/tau
- **功能**：中文轉台語羅馬拼音
- **標音系統**：台羅拼音 (Tâi-lô)
- **準確度**：專業台語標音，涵蓋台語特殊音素

### TTS引擎架構

#### 主要方案：意傳科技TTS API

**技術特色**：
- **真正台語語音**：非中文TTS的台語念法
- **自然語調**：符合台語語音特色
- **高品質合成**：專業級語音品質
- **穩定性修復**：已解決404錯誤音檔捕獲問題

**API詳情**：
- 整段語音合成：完整句子處理
- 單詞語音合成：個別詞彙處理  
- 智能檔名清理：支援特殊字符處理
- 音檔格式：WAV (16kHz, 16bit)

#### 備用方案1：Windows SAPI TTS

**適用情境**：
- 意傳科技API暫時無法使用
- 網路連線不穩定
- 需要快速語音輸出

**技術實現**：
```python
import win32com.client
sapi = win32com.client.Dispatch("SAPI.SpVoice")
```

#### 備用方案2：Google TTS

**特色**：
- 雲端語音合成
- 多語言支援
- 高可用性保證
- 作為最終備用方案

## 故障排除

### 常見問題與解決方案

#### 1. 語音辨識失敗
**症狀**：錄音後無法識別內容
**解決方案**：
- 確認麥克風權限已開啟
- 檢查網路連線狀態
- 嘗試更清楚的發音
- 確認FFmpeg正常安裝

#### 2. LLM對話無回應
**症狀**：AI不回應或回應錯誤
**解決方案**：
```bash
# 檢查Ollama服務狀態
ollama list
ollama serve

# 重新下載模型
ollama pull gemma2:2b
```

#### 3. TTS語音合成失敗
**症狀**：無法播放AI回應語音
**解決方案**：
- 檢查意傳科技API連線狀態
- 系統會自動切換到備用TTS引擎
- 確認Windows SAPI功能正常

#### 4. 網頁界面異常
**症狀**：按鈕無反應或界面錯亂
**解決方案**：
- 清除瀏覽器緩存
- 確認JavaScript已啟用
- 檢查瀏覽器相容性 (推薦Chrome/Edge)

### 系統監控

應用程式提供詳細的日誌輸出：

```bash
python app.py
# 會顯示：
# 啟動台語語音對話 Web 應用程式
# 整合意傳科技 API（基於 TauPhahJi-BangTsam 規範）
# 已修復404錯誤音檔捕獲問題
# API 文檔來源: TauPhahJi-API-docs/API和組件文檔.md
```

## 效能與優化

### 硬體需求

**最低需求**：
- CPU：Intel i3 或 AMD等效
- RAM：4GB
- 儲存：500MB可用空間 (大幅縮減!)
- 網路：穩定寬頻連線

**推薦配置**：
- CPU：Intel i5 或 AMD Ryzen 5
- RAM：8GB以上
- 儲存：SSD硬碟
- 網路：光纖寬頻

### 專案輕量化成果

| 優化項目 | 簡化前 | 簡化後 | 節省 | 改善 |
|----------|--------|--------|------|------|
| **專案總體積** | ~104MB | <1MB | 99.9% | 極速啟動 |
| **檔案數量** | 24,000+ | <50 | 99.8% | 維護簡單 |
| **核心檔案** | 分散 | 集中 | N/A | 部署便利 |
| **啟動時間** | 10-15秒 | 2-3秒 | 70% | 即時響應 |

### 軟體優化

1. **模型選擇**：根據硬體能力選擇合適的LLM模型
2. **緩存管理**：自動清理static/uploads目錄
3. **API連線**：智能重試機制與連線池
4. **記憶體管理**：優化的資源使用監控

## 貢獻指南

歡迎參與專案改進！

### 貢獻方式

1. **回報問題**：在GitHub Issues提出錯誤報告
2. **功能建議**：提出新功能或改進建議  
3. **程式碼貢獻**：提交Pull Request
4. **文檔改進**：協助完善說明文件

### 開發環境設定

```bash
# Clone專案
git clone [repository-url]
cd tai5-uan5-gian5-gi2-kang1-ku7

# 建立開發環境
python -m venv dev_env
dev_env\Scripts\activate  # Windows
# source dev_env/bin/activate  # Linux/Mac

# 安裝開發依賴
pip install -r requirements.txt
```

## 授權與致謝

### 開源授權
本專案採用 **MIT License** 授權條款，詳見 [LICENSE](LICENSE) 檔案。

### 致謝名單

- **意傳科技 (iTaigi)**：提供優質的台語標音與TTS API服務
- **國立台南大學 (NUTN)**：台語專門Whisper模型開發
- **Ollama團隊**：優秀的本地LLM推理引擎
- **OpenAI Whisper**：強大的語音辨識基礎模型
- **TauPhahJi-BangTsam專案**：完整的台語處理API規範

### 相關專案

- [TauPhahJi-BangTsam](https://github.com/i3thuan5/TauPhahJi-BangTsam)：台語處理工具集
- [Whisper-Taiwanese-model](https://huggingface.co/NUTN-KWS/Whisper-Taiwanese-model-v0.5)：台語語音辨識模型

---

## 專案願景

**打造最優質且最輕量的台語語音對話體驗，讓AI也能說一口流利的台語！**

透過整合最新的AI技術與台語專門模型，並採用極致精簡的設計理念，我們致力於：
- 推廣台語數位化與現代化應用
- 支援台語教育與學習  
- 促進台語語音技術研究發展
- 建立台語AI生態系統
- 提供輕量化的AI解決方案典範

**一起為台語的數位未來努力！精簡即是美！**
