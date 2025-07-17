# 台語語音對話系統

專門為台語語音辨識與智能對話打造的Web應用系統，整合多項先進技術提供流暢的台語語音互動體驗。

## 核心特色

- **專業台語支援**：使用專門訓練的台語語音模型，真正理解台語語音特色
- **智能對話**：整合大型語言模型，提供自然流暢的對話體驗  
- **自訓練TTS引擎**：使用自訓練的遠端TTS服務，確保語音輸出品質
- **現代化介面**：響應式Web界面，支援桌面與行動裝置
- **即時處理**：毫秒級語音辨識與合成，接近真實對話體驗
- **超輕量化**：精簡專案結構，啟動更快、部署更簡便
- **智能格式轉換**：自動處理羅馬拼音格式轉換，確保TTS品質

## 系統架構

### 完整處理流程
```
台語語音輸入 → STT辨識 → LLM智能對話 → 台語標音 → 格式轉換 → TTS語音合成 → 台語語音輸出
     ↓           ↓          ↓           ↓         ↓          ↓           ↓
  .webm錄音   中文文字   智能中文回應  台語羅馬拼音  數字調格式  .wav音檔   台語語音播放
```

### 技術堆疊
- **前端**：HTML5 + JavaScript + Web Audio API
- **後端**：Flask + Python 3.8+
- **STT引擎**：Whisper (台語專門模型 NUTN-KWS/Whisper-Taiwanese-model-v0.5)
- **LLM對話**：Ollama + Gemma/Llama3系列
- **標音服務**：意傳科技標音API
- **TTS引擎**：自訓練遠端TTS (透過`config.py`或`.env`設定)
- **格式轉換**：自研羅馬拼音轉換器

## 專案結構

```
台語語音對話系統/ 
├── app.py                          # 主應用程式 (Flask Web服務)
├── config.py                       # 系統配置文件
├── remote_tts_service.py           # 遠端TTS服務模組
├── romanization_converter.py      # 羅馬拼音格式轉換器
├── requirements.txt                # Python依賴套件清單
├── cleanup.ps1                     # 專案清理腳本
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
├── ffmpeg_bin/                   # 音訊處理工具
│   └── ffmpeg.exe               # FFmpeg二進位檔案
│
└── TauPhahJi-API-docs/           # 意傳科技API文檔
    ├── API和組件文檔.md          # API規範文檔
    ├── 專案說明書.md             # 專案說明
    ├── 專案完整說明書.md         # 完整技術文檔
    ├── 狀態管理和部署指南.md     # 部署指南
    ├── README.md                # 專案介紹
    └── LICENSE                  # 授權條款
```

## 最新功能特色

### 自訓練遠端TTS服務

#### 技術特色：
- **專業台語語音**：基於 SuiSiann-HunLian 模型
- **高品質合成**：針對台語語音特色優化
- **數字調格式**：支援 `gua2 si7 ko3` 格式輸入
- **穩定連線**：90秒超時設定，確保長句處理

#### API詳情：
- **端點**：(透過`config.py`或`.env`設定)
- **參數**：`taibun` (數字調格式文字)
- **回應**：MP3音檔 (自動轉換為WAV)
- **檔名格式**：`static/remote_tts_{text}_{timestamp}.wav`

### 智能羅馬拼音轉換器

#### 格式轉換功能：
- **輸入格式**：意傳科技API羅馬拼音 `guá sī kò sió tsōo-tshiú`
- **輸出格式**：數字調格式 `gua2 si7 ko3 sio2 tsoo7-tshiu2`
- **Unicode支援**：正確處理組合字符 `a̍` → `a8`
- **智能判斷**：自動識別已轉換格式，避免重複處理

#### 聲調映射規則：
- **第2聲**：`á é í ó ú` → `a2 e2 i2 o2 u2`
- **第3聲**：`à è ì ò ù` → `a3 e3 i3 o3 u3`
- **第5聲**：`â ê î ô û` → `a5 e5 i5 o5 u5`
- **第7聲**：`ā ē ī ō ū` → `a7 e7 i7 o7 u7`
- **第8聲**：`a̍ e̍ i̍ o̍ u̍` → `a8 e8 i8 o8 u8`

### 專案維護工具

#### 自動清理腳本 (`cleanup.ps1`)：
- **Python快取清理**：自動刪除 `__pycache__` 和 `.pyc` 檔案
- **日誌檔案管理**：清理 `app.log` 和其他日誌檔案
- **音檔空間管理**：刪除重複測試音檔和舊檔案
- **上傳檔案清理**：自動清理3天前的上傳檔案
- **統計報告**：顯示清理後的檔案統計

## 快速開始

### 前置需求

1. **Python 3.8+** 環境
2. **網路連線** (存取意傳科技API和遠端TTS)
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
- `soundfile` - 音訊檔案處理

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

#### 3. 設定FFmpeg (音訊處理工具)

```bash
# 下載FFmpeg並放置到專案目錄
# 1. 訪問 https://ffmpeg.org/download.html
# 2. 下載Windows版本
# 3. 解壓縮後將ffmpeg.exe放到 ffmpeg_bin/ 目錄中
# 4. 或使用Chocolatey: choco install ffmpeg
```

#### 4. 啟動應用程式

```bash
cd Taiwanese-Voice-Assistant
python app.py
```

系統會自動：
- 啟動Flask Web服務器 (端口5000)
- 連接Ollama LLM服務
- 初始化遠端TTS服務 (163.13.202.125:5000)
- 初始化羅馬拼音轉換器
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

5. **格式轉換處理**
   - 自動判斷羅馬拼音格式
   - 轉換為數字調格式 (如需要)
   - 確保TTS引擎相容性

6. **台語語音合成**
   - 使用自訓練遠端TTS服務
   - 自動播放生成的台語語音

### 測試功能

#### API測試端點：
- **基本功能測試**：`http://localhost:5000/test_api`
- **遠端TTS測試**：`http://localhost:5000/test_remote_tts?text=li2%20ho2`
- **帶參數測試**：`http://localhost:5000/test_remote_tts?text=gua2%20si7&version=1`

## 技術規格詳解

### STT (語音轉文字)

**模型資訊**：
- **名稱**：NUTN-KWS/Whisper-Taiwanese-model-v0.5
- **類型**：台語專門微調的Whisper模型  
- **訓練數據**：台語語音資料集
- **支援格式**：WebM, WAV, MP3
- **辨識精度**：針對台語優化，顯著優於通用模型
- **語言設定**：正確設定為台語識別

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

### 自訓練遠端TTS服務

**技術特色**：
- **專業台語語音**：基於 SuiSiann-HunLian 模型
- **高品質合成**：針對台語語音特色優化
- **數字調格式**：原生支援 `li2 ho2` 格式
- **穩定連線**：90秒超時設定，支援長句處理

**API詳情**：
- **端點**：http://163.13.202.125:5000/bangtsam
- **參數**：`taibun` (數字調格式文字)
- **回應**：MP3音檔 (自動轉換為WAV)
- **檔名格式**：`static/remote_tts_{text}_{timestamp}.wav`

### 羅馬拼音轉換器

**核心功能**：
- **格式檢測**：自動判斷輸入格式
- **Unicode處理**：正確處理組合字符
- **聲調映射**：精確的台語聲調轉換
- **錯誤處理**：轉換失敗時返回原文

**轉換範例**：
```python
# 輸入：意傳科技API格式
input_text = "guá sī kò sió tsōo-tshiú"

# 輸出：數字調格式
output_text = "gua2 si7 ko3 sio2 tsoo7-tshiu2"

# 特殊字符處理
"tsia̍h-pn̄g" → "tsiah8-png7"
```

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

#### 3. 遠端TTS連線失敗
**症狀**：無法連接到自訓練TTS服務
**解決方案**：
- 檢查 `config.py` 或 `.env` 中的 `REMOTE_TTS_HOST` 和 `REMOTE_TTS_PORT` 設定是否正確
- 檢查網路防火牆設定
- 確認遠端TTS服務正常運行

#### 4. 羅馬拼音轉換錯誤
**症狀**：音檔聽起來奇怪或不正確
**解決方案**：
- 檢查轉換器日誌輸出
- 確認輸入格式正確
- 測試轉換功能：
```python
from romanization_converter import RomanizationConverter
converter = RomanizationConverter()
result = converter.convert_to_numeric_tone("lí hó")
print(result)  # 應該輸出: li2 ho2
```

#### 5. 音檔播放問題
**症狀**：生成的音檔無法播放
**解決方案**：
- 檢查音檔格式是否正確
- 確認瀏覽器支援音檔格式
- 檢查音檔檔案大小是否正常
- 測試音檔下載功能

### 系統監控

應用程式提供詳細的日誌輸出：

```bash
python app.py
# 會顯示：
# 啟動台語語音對話 Web 應用程式
# 使用自訓練遠端 TTS 服務 (163.13.202.125:5000)
# 整合意傳科技標音 API
# 初始化自訓練遠端TTS服務...
# 遠端TTS服務初始化成功
# 初始化羅馬拼音格式轉換器...
# 羅馬拼音轉換器初始化成功
# 系統初始化完成！
```

## 專案維護

### 自動清理腳本

使用 PowerShell 清理腳本：

```powershell
# 執行清理腳本
.\cleanup.ps1

# 會自動清理：
# - Python 快取檔案 (__pycache__, *.pyc)
# - 日誌檔案 (*.log)
# - 重複測試音檔 (*li2_ho2*)
# - 舊的上傳檔案 (>3天)
# - 舊的音檔 (>7天)
```

### 手動維護

```bash
# 清理Python快取
Remove-Item -Recurse -Force __pycache__

# 清理測試音檔
Get-ChildItem static\ | Where-Object { $_.Name -like "*test*" } | Remove-Item

# 清理舊上傳檔案
Get-ChildItem uploads\ | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-3) } | Remove-Item
```

## 效能與優化

### 硬體需求

**最低需求**：
- CPU：Intel i3 或 AMD等效
- RAM：4GB
- 儲存：1GB可用空間
- 網路：穩定寬頻連線

**推薦配置**：
- CPU：Intel i5 或 AMD Ryzen 5
- RAM：8GB以上
- 儲存：SSD硬碟
- 網路：光纖寬頻

### 專案優化成果

| 優化項目 | 改善內容 | 效果 |
|----------|----------|------|
| **TTS引擎** | 自訓練遠端TTS | 高品質台語語音 |
| **格式轉換** | 智能羅馬拼音轉換 | 音檔品質提升 |
| **專案結構** | 模組化設計 | 維護便利 |
| **錯誤處理** | 完整的錯誤捕獲 | 用戶體驗佳 |
| **清理工具** | 自動化維護 | 空間管理 |

### 軟體優化

1. **TTS服務**：專門的自訓練遠端TTS確保最佳語音品質
2. **格式轉換**：自動判斷避免重複處理
3. **緩存管理**：自動清理static/uploads目錄
4. **API連線**：智能重試機制與連線池
5. **記憶體管理**：優化的資源使用監控

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
cd Taiwanese-Voice-Assistant

# 建立開發環境
python -m venv dev_env
dev_env\Scripts\activate  # Windows
# source dev_env/bin/activate  # Linux/Mac

# 安裝開發依賴
pip install -r requirements.txt

# 執行測試
python -c "from romanization_converter import RomanizationConverter; RomanizationConverter().test_conversion()"
```

## 授權與致謝

### 開源授權
本專案採用 **MIT License** 授權條款，詳見 [LICENSE](LICENSE) 檔案。

### 致謝名單

- **意傳科技 (iTaigi)**：提供優質的台語標音API服務
- **國立台南大學 (NUTN)**：台語專門Whisper模型開發
- **Ollama團隊**：優秀的本地LLM推理引擎
- **OpenAI Whisper**：強大的語音辨識基礎模型
- **TauPhahJi-BangTsam專案**：完整的台語處理API規範
- **SuiSiann-HunLian**：自訓練TTS模型基礎

### 相關專案

- [TauPhahJi-BangTsam](https://github.com/i3thuan5/TauPhahJi-BangTsam)：台語處理工具集
- [Whisper-Taiwanese-model](https://huggingface.co/NUTN-KWS/Whisper-Taiwanese-model-v0.5)：台語語音辨識模型
- [Ollama](https://ollama.ai/)：本地LLM推理引擎

---

## 專案願景

**打造最優質且最智能的台語語音對話體驗，讓AI也能說一口流利的台語！**

透過整合最新的AI技術與台語專門模型，並採用自訓練遠端TTS引擎和智能格式轉換，我們致力於：
- 推廣台語數位化與現代化應用
- 支援台語教育與學習  
- 促進台語語音技術研究發展
- 建立台語AI生態系統
- 提供高品質的台語語音合成體驗

**一起為台語的數位未來努力！讓技術服務文化傳承！**

## 更新日誌

### v2.1.0 (最新版本)
- **安全性提升**：移除所有硬編碼的IP地址和敏感信息
- **配置系統**：新增 `config.py`，支援透過環境變數或文件進行配置
- **啟動檢查**：應用程式啟動時會檢查遠端TTS服務是否配置
- **README更新**：更新文檔，說明如何配置系統

### v2.0.0
- 整合自訓練遠端TTS服務
- 新增智能羅馬拼音格式轉換器
- 修正Unicode組合字符處理問題
- 新增專案清理工具 (cleanup.ps1)
- 優化錯誤處理和日誌輸出
- 更新完整的API測試端點
- 移除表情符號，採用純文字介面

### v1.0.0 (基礎版本)
- 台語語音辨識功能
- LLM智能對話
- 意傳科技API整合
- 基本TTS功能
- Web界面實現
