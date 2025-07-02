# 鬥拍字 - API文檔和組件文檔補充

## 🌐 API文檔詳解

### API服務架構概覽
專案依賴意傳科技提供的兩個主要API服務：

| API服務 | 端點 | 功能 | 狀態 |
|---------|------|------|------|
| 台語標音API | hokbu.ithuan.tw | 文字轉標音 | ✅ 活躍 |
| 語音合成API | hapsing.ithuan.tw | TTS語音生成 | ✅ 活躍 |

### 🔍 標漢字音標 API

#### 基本資訊
- **端點**: `https://hokbu.ithuan.tw/tau`
- **方法**: POST
- **內容類型**: application/x-www-form-urlencoded
- **用途**: 將台語文字轉換為羅馬拼音標音

#### 請求格式
```javascript
// API調用範例
const 查詢標音 = async (台語文字) => {
  const response = await superagent
    .post('https://hokbu.ithuan.tw/tau')
    .type('form')
    .send({
      taibun: 台語文字.trim()
    });
  
  return response.body;
};
```

#### 請求參數
| 參數名 | 類型 | 必填 | 說明 | 範例 |
|--------|------|------|------|------|
| taibun | string | ✅ | 台語文字輸入 | "逐家tsò-hué來chhit4-tho5！" |

#### 回應格式
```json
{
  "分詞": "逐家 tsò-hué 來 chhit4-tho5 ！",
  "kiatko": [
    {
      "分詞": "逐家",
      "漢字": "逐家",
      "KIP": "ta̍k-ke"
    },
    {
      "分詞": "tsò-hué",
      "漢字": "tsò-hué", 
      "KIP": "tsò-hué"
    },
    {
      "分詞": "來",
      "漢字": "來",
      "KIP": "lâi"
    },
    {
      "分詞": "chhit4-tho5",
      "漢字": "chhit4-tho5",
      "KIP": "chhit-thô"
    },
    {
      "分詞": "！",
      "漢字": "！",
      "KIP": "！"
    }
  ]
}
```

#### 回應欄位說明
| 欄位名 | 類型 | 說明 |
|--------|------|------|
| 分詞 | string | 整句的分詞結果 |
| kiatko | Array | 每個詞的詳細資訊陣列 |
| kiatko[].分詞 | string | 原始分詞 |
| kiatko[].漢字 | string | 漢字表示 |
| kiatko[].KIP | string | 羅馬拼音標音 |

### 🎵 語音合成 API

#### 整段語音合成
- **端點**: `https://hokbu.ithuan.tw/語音合成`
- **方法**: GET
- **用途**: 生成整句台語語音

#### 請求參數
| 參數名 | 類型 | 必填 | 說明 | 範例 |
|--------|------|------|------|------|
| 查詢腔口 | string | ✅ | 語音腔調 | "閩南語" |
| 查詢語句 | string | ✅ | 分詞後的文字 | "ta̍k-ke tsò-hué lâi chhit-thô" |

#### URL生成範例
```javascript
const 生成整段語音URL = (腔口, 分詞) => {
  return encodeURI(
    `https://hokbu.ithuan.tw/語音合成?查詢腔口=${腔口}&查詢語句=${分詞}`
  );
};

// 使用範例
const audioURL = 生成整段語音URL("閩南語", "ta̍k-ke tsò-hué lâi");
```

#### 單詞語音合成
- **端點**: `https://hapsing.ithuan.tw/bangtsam`
- **方法**: GET
- **用途**: 生成單個詞語的語音

#### 請求參數
| 參數名 | 類型 | 必填 | 說明 | 範例 |
|--------|------|------|------|------|
| taibun | string | ✅ | 單詞羅馬拼音 | "ta̍k-ke" |

#### URL生成範例
```javascript
const 生成單詞語音URL = (羅馬拼音) => {
  return encodeURI(
    `https://hapsing.ithuan.tw/bangtsam?taibun=${encodeURIComponent(羅馬拼音)}`
  );
};

// 使用範例
const wordAudioURL = 生成單詞語音URL("ta̍k-ke");
```

### ⚠️ API使用限制和最佳實踐

#### 使用限制
- **頻率限制**: 每IP每分鐘最多3次音檔下載
- **文字長度**: 建議單次查詢不超過200字
- **同時請求**: 避免同時發送多個請求

#### 錯誤處理
```javascript
const 安全API調用 = async (台語文字) => {
  try {
    const 結果 = await 查詢標音(台語文字);
    return { 成功: true, 資料: 結果 };
  } catch (錯誤) {
    console.error('API調用失敗:', 錯誤);
    
    if (錯誤.status === 429) {
      return { 成功: false, 訊息: '請求過於頻繁，請稍後再試' };
    } else if (錯誤.status === 500) {
      return { 成功: false, 訊息: '服務器暫時無法使用' };
    } else {
      return { 成功: false, 訊息: '網路連線錯誤' };
    }
  }
};
```

---

## 🧩 React組件文檔

### 組件層次結構圖

```
MyProvider (應用程式根組件)
└── Router (路由管理)
    └── 網站 (主要布局)
        ├── 頁頭 (網站頭部)
        ├── 查 (查詢頁面)
        │   ├── 查表格 (輸入表格)
        │   └── 翻譯結果 (結果顯示)
        │       ├── ButtonStack (操作按鈕群)
        │       └── 漢羅列表 (結果列表)
        │           └── PlayButton, DownloadButton (音頻控制)
        └── 頁尾 (網站底部)
```

### 🔗 MyProvider (根組件)

#### 檔案位置
`src/providers/index.jsx`

#### 功能說明
- 應用程式的最頂層組件
- 配置Redux Store和中間件
- 設置React Router路由
- 提供全域狀態管理

#### 程式碼結構
```jsx
const MyProvider = () => {
  const store = getAppStore(); // 創建Redux Store
  
  return (
    <Provider store={store}>
      <Router history={browserHistory}>
        <Route path='/' component={網站}>
          <IndexRoute component={查}/>
          <Route path='%E8%AC%9B/:khiunn/:ku' component={查}/>
          <Route path='%E8%AC%9B(/:ku)' component={查}/>
          <Route path='%E8%AC%9B' component={查}/>
          <Route path='**/:ku' component={查}/>
        </Route>
      </Router>
    </Provider>
  );
};
```

#### Store配置
```javascript
const getAppStore = () => {
  const middlewares = [thunk];
  
  // 開發環境添加Logger
  if (process.env.NODE_ENV !== "production") {
    middlewares.push(createLogger);
  }
  
  return createStore(
    reducer,
    applyMiddleware(...middlewares)
  );
};
```

### 🏠 網站 (主布局組件)

#### 檔案位置
`src/網站/網站.jsx`

#### Props介面
```typescript
interface 網站Props {
  params: {
    ku?: string;      // 查詢語句參數
    khiunn?: string;  // 腔調參數
  };
  children: React.ReactNode; // 子組件
}
```

#### 功能特色
- 提供整體頁面布局結構
- 自動解析URL參數並傳遞給子組件
- 使用demo-ui的Layout組件
- 支援預設值處理

#### 程式碼範例
```jsx
class 網站 extends React.Component {
  render() {
    const { ku, khiunn } = this.props.params;
    
    return (
      <Layout>
        <頁頭/>
        {React.cloneElement(
          this.props.children,
          {
            語句: ku || config.範例查詢(),
            腔: khiunn || config.預設腔口(),
          }
        )}
        <頁尾/>
      </Layout>
    );
  }
}
```

### 🔍 查 (查詢頁面)

#### 檔案位置
`src/頁/查/查.jsx`

#### 功能說明
- 查詢功能的主要頁面組件
- 整合查詢表格和結果顯示
- 管理查詢相關的UI狀態

### 📝 查表格 (查詢輸入組件)

#### 檔案位置
- 組件: `src/頁/查/查表格.jsx`
- 容器: `src/頁/查/查表格.container.js`

#### 功能特色
- 提供台語文字輸入介面
- 處理表單提交和驗證
- 支援範例文字預填
- 響應式輸入體驗

### 📊 翻譯結果 (結果顯示組件)

#### 檔案位置
- 組件: `src/元素/翻譯/翻譯結果.jsx`
- 容器: `src/元素/翻譯/翻譯結果.container.js`

#### Props介面
```typescript
interface 翻譯結果Props {
  腔口: string;           // 查詢腔調
  正在查詢: boolean;      // 載入狀態
  發生錯誤: boolean;      // 錯誤狀態
  分詞: string;           // 分詞結果
  綜合標音: Array<{       // 標音結果陣列
    分詞: string;
    漢字: string;
    KIP: string;
  }>;
}
```

#### 核心方法
```jsx
class 翻譯結果 extends React.Component {
  // 生成複製按鈕群組
  取得複製鈕群() {
    const { 綜合標音 } = this.props;
    const 複製內容 = 計算複製內容(綜合標音);
    
    return Object.keys(複製內容).map(key => (
      <CopyButton 
        key={key} 
        複製內容={複製內容[key]} 
        按鈕名={key}
      />
    ));
  }
  
  // 生成整段音頻按鈕
  取得整段鈕群() {
    const { 綜合標音, 分詞, 腔口 } = this.props;
    
    if (綜合標音.length > 0) {
      const src = 意傳服務.語音合成({ 腔口, 分詞 });
      
      return (
        <Block>
          <DownloadButton src={src}>整段下載</DownloadButton>
          <PlayButton src={src}>整段播放</PlayButton>
        </Block>
      );
    }
    return null;
  }
}
```

### 📋 漢羅列表 (結果列表組件)

#### 檔案位置
- 組件: `src/元素/顯示/漢羅列表.jsx`
- 容器: `src/元素/顯示/漢羅列表.container.js`

#### Props介面
```typescript
interface 漢羅列表Props {
  結果腔口: string;       // 結果腔調
  綜合標音: Array<{       // 標音資料
    KIP: string;          // 羅馬拼音
    漢字: string;         // 漢字
    分詞: string;         // 分詞
  }>;
}
```

#### 渲染邏輯
```jsx
class 漢羅列表 extends React.Component {
  render() {
    const { 綜合標音 } = this.props;
    
    return (
      <div>
        <p>（因資源有限，1 IP 1分鐘內上 tsē 下載 3 句音檔。）</p>
        {綜合標音.map((綜音, i) => {
          const src = encodeURI(
            "https://hapsing.ithuan.tw/bangtsam?taibun=" +
            encodeURIComponent(綜音.KIP)
          );
          
          return (
            <div key={i}>
              <PlayButton src={src}/>
              <DownloadButton src={src}/>
              <ruby className="app ruby">
                {綜音.漢字}
                <rt>{綜音.KIP}</rt>
              </ruby>
            </div>
          );
        })}
      </div>
    );
  }
}
```

### 🔧 工具函數組件

#### 複製功能 (src/utils/複製.js)
```javascript
const 計算複製內容 = (綜合標音 = []) => {
  if (!綜合標音 || 綜合標音.length < 1) {
    return 綜合標音;
  }

  return 綜合標音
    .map((item) => {
      const 漢字 = item.漢字.replace(/ /g, "");
      return {
        漢字羅馬: [漢字, item.KIP].join("\n"),
        羅馬字: item.KIP,
        漢字,
        分詞: item.分詞,
      };
    })
    .reduce((acc, item) => ({
      漢字羅馬: [acc.漢字羅馬, item.漢字羅馬].join("\n"),
      漢字: [acc.漢字, item.漢字].join("\n"),
      羅馬字: [acc.羅馬字, item.羅馬字].join("\n"),
      分詞: [acc.分詞, item.分詞].join("\n"),
    }));
};
```

### 📱 容器組件模式

#### 設計原則
- **分離關注點**: 容器負責狀態，組件負責顯示
- **可重用性**: 組件可以在不同容器中使用
- **測試友善**: 純組件易於測試

#### 典型容器結構
```javascript
// 範例: 翻譯結果.container.js
import { connect } from "react-redux";
import { 查詢語句 } from "../../actions";
import 翻譯結果 from "./翻譯結果";

const mapStateToProps = (state) => ({
  腔口: state.查詢.腔口,
  正在查詢: state.查詢.正在查詢,
  發生錯誤: state.查詢.發生錯誤,
  分詞: state.查詢結果.分詞,
  綜合標音: state.查詢結果.綜合標音,
});

const mapDispatchToProps = {
  查詢語句,
};

export default connect(
  mapStateToProps, 
  mapDispatchToProps
)(翻譯結果);
```

---

## 🎯 組件使用指南

### 新增自定義組件

#### 1. 創建組件文件
```jsx
// src/元素/新組件/新組件.jsx
import React from "react";
import PropTypes from "prop-types";

class 新組件 extends React.Component {
  render() {
    const { 屬性1, 屬性2 } = this.props;
    
    return (
      <div>
        <h2>{屬性1}</h2>
        <p>{屬性2}</p>
      </div>
    );
  }
}

新組件.propTypes = {
  屬性1: PropTypes.string.isRequired,
  屬性2: PropTypes.string,
};

export default 新組件;
```

#### 2. 創建容器文件 (如需要)
```javascript
// src/元素/新組件/新組件.container.js
import { connect } from "react-redux";
import 新組件 from "./新組件";

const mapStateToProps = (state) => ({
  屬性1: state.某個狀態.屬性1,
  屬性2: state.某個狀態.屬性2,
});

export default connect(mapStateToProps)(新組件);
```

#### 3. 在父組件中使用
```jsx
import Container新組件 from "./元素/新組件/新組件.container";

// 在render方法中
<Container新組件 />
```

### 組件最佳實踐

#### PropTypes定義
```jsx
// 完整的PropTypes範例
組件名稱.propTypes = {
  // 必填字串
  標題: PropTypes.string.isRequired,
  
  // 可選字串
  描述: PropTypes.string,
  
  // 數字
  數量: PropTypes.number,
  
  // 布林值
  顯示: PropTypes.bool,
  
  // 函數
  onClick: PropTypes.func,
  
  // 陣列
  項目列表: PropTypes.array,
  
  // 物件陣列
  資料列表: PropTypes.arrayOf(PropTypes.shape({
    id: PropTypes.number,
    名稱: PropTypes.string,
  })),
  
  // 子組件
  children: PropTypes.node,
};
```

#### 狀態管理最佳實踐
```jsx
class 狀態組件 extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      載入中: false,
      錯誤訊息: null,
      資料: null,
    };
  }
  
  // 使用箭頭函數避免this綁定問題
  處理點擊 = () => {
    this.setState({ 載入中: true });
    // 處理邏輯
  }
  
  // 生命週期方法
  componentDidMount() {
    // 組件載入後執行
  }
  
  componentWillUnmount() {
    // 組件卸載前清理
  }
}
``` 