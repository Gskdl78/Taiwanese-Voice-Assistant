# SuiSiann-HunLian 臺語語音合成 (TTS) API 使用指南

這份文件說明如何在遠端主機上啟動 SuiSiann-HunLian 的語音合成服務，並透過 API 在本地端或其他服務中靈活使用。

---

## 1. 在遠端主機上啟動服務

首先，您需要在已經完成模型訓練的遠端主機上，將 TTS 模型以 API 服務的形式啟動。這個步驟只需要執行一次。

### 步驟 1: 進入專案目錄

透過 SSH 連線到您的遠端主機，並切換到存放 `docker-compose.yaml` 的正確目錄。

```bash
cd ~/SuiSiann-HunLian-main/fatchord-WaveRNN
```

### 步驟 2: 啟動 Docker 服務

使用 `docker-compose` 指令來啟動所有相關的容器，包含 Nginx 網頁伺服器和後端的 TTS 核心服務。`-d` 參數會讓服務在背景持續運行。

```bash
docker-compose up -d --force-recreate
```
*   `--force-recreate` 參數可以確保容器使用最新的設定檔來建立。

### 步驟 3: 確認服務狀態

執行以下指令，檢查服務是否都正常運行。

```bash
docker ps
```
您應該會看到 `fatchord-wavernn_hapsing_1` 和 `fatchord-wavernn_hokbu-nginx_1` 等容器的 `STATUS` 顯示為 `Up`。

---

## 2. 如何使用 API 進行語音合成

當遠端服務啟動後，您就可以在**任何地方**（例如您的本地電腦）透過呼叫 API 來合成語音。

**重要：** 在所有範例中，請將 `<遠端主機IP>` 替換為您遠端伺服器的真實 IP 位址（例如 `163.13.202.125`）。

### API 參數說明
-   **API 端點 (Endpoint)**: `/bangtsam` (直接下載音檔) 或 `/hapsing` (回傳音檔資訊的 JSON)
-   **文字參數**: `taibun`
-   **參數格式**: 使用臺羅數字調，音節之間用空格分開。

### 方法 A: 使用網頁瀏覽器 (最簡單)

這是最直覺的測試方法，可以直接在瀏覽器下載合成好的 `.mp3` 檔。

1.  打開您本地的 Chrome、Edge 或任何網頁瀏覽器。
2.  在網址列輸入以下格式的網址，然後按下 Enter。

    **範例：合成 "lí hó" (li2 ho2)**
    ```
    http://<遠端主機IP>:5000/bangtsam?taibun=li2%20ho2
    ```

    **範例：合成 "逐家好，來創啥物！" (tak10-ke7 tsə2-hue1 lai7 tsʰit8-tʰə5 !)**
    ```
    http://<遠端主機IP>:5000/bangtsam?taibun=tak10-ke7%20ts%C9%992-hue1%20lai7%20ts%CA%B0it8-t%CA%B0%C9%955%20!
    ```
    瀏覽器會自動處理特殊字元的編碼，並開始下載 `taiuanue.mp3`。

### 方法 B: 使用 cURL 指令 (適合在終端機操作)

如果您習慣使用命令列，可以在您本地的 PowerShell 或 Terminal 中執行。

**範例：合成 "lí hó" 並存成 `li-ho.mp3`**
```powershell
# 在 PowerShell 或 Terminal 中執行
curl "http://<遠端主機IP>:5000/bangtsam?taibun=li2%20ho2" -o li-ho.mp3
```

**範例：合成帶有特殊符號的長句並存成 `test.mp3`**
> **注意：** 為避免特殊字元在終端機中的編碼問題，建議使用 `--data-urlencode` 參數，這是最穩定的 cURL 使用方式。

```powershell
curl "http://<遠端主機IP>:5000/bangtsam" --data-urlencode "taibun=tak10-ke7 tsə2-hue1 lai7 tsʰit8-tʰə5 !" -o test.mp3
```

### 方法 C: 使用 Python 腳本 (最穩定、最推薦)

如果您想在自己的應用程式中整合 TTS 功能，使用 Python 腳本是最佳選擇。`requests` 函式庫會完美處理所有 URL 編碼細節。

1.  確保您已安裝 `requests` 函式庫: `pip install requests`
2.  建立一個 Python 檔案 (例如 `call_tts.py`) 並貼上以下程式碼：

```python
import requests

# --- 設定 ---
REMOTE_IP = "<遠端主機IP>"  # <-- 請務必修改這裡
TEXT_TO_SYNTHESIZE = "tak10-ke7 tsə2-hue1 lai7 tsʰit8-tʰə5 !"
OUTPUT_FILENAME = "output.mp3"
# --- 設定結束 ---

# 組合 API 網址和參數
api_url = f"http://{REMOTE_IP}:5000/bangtsam"
params = {"taibun": TEXT_TO_SYNTHESIZE}

print(f"正在請求合成文字: '{TEXT_TO_SYNTHESIZE}'")

try:
    # 發出請求，requests 函式庫會自動處理好 URL 編碼
    response = requests.get(api_url, params=params, timeout=90) # 加長超時時間

    # 檢查請求是否成功
    if response.status_code == 200:
        # 將收到的音檔內容寫入檔案
        with open(OUTPUT_FILENAME, 'wb') as f:
            f.write(response.content)
        print(f"成功！音檔已儲存為 {OUTPUT_FILENAME}")
    else:
        # 如果伺服器回傳錯誤
        print(f"錯誤！伺服器回應狀態碼: {response.status_code}")
        print(f"錯誤訊息: {response.text}")

except requests.exceptions.RequestException as e:
    print(f"連線錯誤！請檢查 IP 位址和網路連線。錯誤詳情: {e}")

```
3.  執行腳本: `python call_tts.py`，合成好的音檔就會出現在同一個目錄下。

---

## 3. 疑難排解

-   **連線被拒絕 (Connection Refused)**:
    -   確認 `docker ps` 中容器是否正常運行。
    -   檢查 `docker-compose.yaml` 中的 `ports` 設定是否為 `"5000:80"`。
-   **找不到頁面 (404 Not Found)**:
    -   確認您使用的 API 路徑是否正確 (應為 `/bangtsam` 或 `/hapsing`)。
    -   確認您使用的參數是否為 `taibun`。
    -   檢查 Nginx 設定檔 (`hokbu-nginx/default.conf`) 是否正確轉發請求。
-   **音檔內容錯誤或無法播放**:
    -   確認 `taibun` 參數中的臺羅拼音格式是否正確。
    -   使用瀏覽器或 Python 腳本進行測試，以排除終端機的編碼問題。
-   **沒有回應或超時 (Timeout)**:
    -   檢查遠端主機的防火牆 (`sudo ufw status`)，確保 5000/tcp 端口是允許 (ALLOW) 的。如果不是，請執行 `sudo ufw allow 5000/tcp`。
    -   檢查遠端服務的日誌 (`docker logs fatchord-wavernn_hapsing_1`)，確認模型是否在載入時或合成中卡住。 