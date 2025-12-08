# AI CUP 2025 秋季賽  
## 電腦斷層主動脈瓣物件偵測競賽

我們參加了 AI CUP 2025 年的電腦斷層主動脈瓣膜物件偵測競賽，以下是 Private 計分上榜模型的復現流程，改良說明及效果請見競賽報告。

## 隊伍計分榜排名

| Stage   | Rank | Score     |
|---------|------|-----------|
| Public  | 17/301 | 0.970913 |
| Private | 12/301 | 0.973514 |

---

## 模型復現流程（Linux + Anaconda）

本專案模型是採用 ultralytics 提供的 YOLO11 架構中 x 等級模型加以改良，以下是最佳模型的復現流程。  
（若有路徑出錯問題，請自行至 YAML 檔內將路徑改成絕對路徑）

---

### 第一步：安裝 Python 環境

```bash
conda create -n valve_detection python==3.11.14 -y
```

### 第二步：切換環境

```bash
conda activate valve_detection
```

### 第三步：移動到專案資料夾

```bash
cd Heart-Valve-Detection-main
```

### 第四步：安裝必要套件

```bash
pip install -r requirements.txt
```

### 第五步：修改 `config_IS_x.yaml`（若要復現則跳過）

內容包含：

- model：本次改良模型路徑  
- out：模型結果比較輸出資料夾  
- name：該次模型訓練名稱  

模型訓練結果會儲存在：

```
./{out}/{name}/
```

---

### 第六步：開始訓練  
（訓練時間 18～24 小時不等，依設備而定）

```bash
python train_IS.py --config config_IS_x.yaml
```

---

### 第七步：修改 `inf_IS_x.yaml`

若要使用以訓練好的模型，請自行下載 

(linux): 
```
cd results
```
```
gdown --fuzzy "https://drive.google.com/file/d/1iT2iCLMsTPpa1EyzFknHvw_DFbtvhUct/view?usp=sharing"
```
```
cd ..
```

(windows):

https://drive.google.com/file/d/1iT2iCLMsTPpa1EyzFknHvw_DFbtvhUct/view



若要使用自行訓練的模型，請將路徑自行修改為：

```
./{out}/{name}/weights/best.pt
```

### 第八步：執行 inference_with_AfterProcessing.py

```bash
python inference_with_AfterProcessing.py --config inf_IS_x.yaml
```

最終結果會輸出在：

```
./results/Final.txt
```

（可自行修改名稱，例如：`XXX.txt`）

---

## YOLO 架構中被修改的部分

以下為 ultralytics YOLO 架構中被調整的檔案：

```
./ultralytics/nn/tasks.py                # 訓練流程
./ultralytics/nn/blocks.py               # SDI 改編模塊
./ultralytics/cfg/models/11/yolo11.yaml  # 模型架構
