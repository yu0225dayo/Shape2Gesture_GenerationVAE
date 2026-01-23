# Shape2Gesture_GenerationModel

接触部位形状を介した、両手把持姿勢の生成モデル

## 研究背景
学部では、「接触部位形状を介した、全体形状と両手把持ジェスチャの相互検索システム」を構築した。<br>
相互検索システムではDBが不可欠であるため、本プロジェクトではDBなしで直接生成できる生成モデルに拡張する。
相互検索システム：https://github.com/yu0225dayo/Shape2Gesture_SearchModel

## 提案手法

### モデル構造
![Model overview](figs/model.png)
物体の接触部位形状から手首座標系の手形状を生成するVAEと、手首座標と手の向きを生成するVAEにモデルを分離させて学習するフレームワークを提案する。<br>
単一Decoderで生成すると、手形状が崩壊する問題に直面したため、学習・生成の安定化を図るため分離する。

### 損失関数(席取りLoss)
![Model overview](figs/abst_sekitori.png)
ボトル形状など対称性のある形状は、把持方向が無数に存在する。あらゆる方向の把持姿勢が生成されることが理想であるため、VAEの潜在空間を拘束する必要がある。<br>
VAEは確率的分布であることから出力にランダム性が生まれる。その領域を拘束するというLossを提案
![Model overview](figs/latentspace.png)

## 主な特徴

- ✅ **分離されたモデル構造**
手形状生成VAEと位置・向き生成VAEに分離し学習・生成の安定化<br>
PointNet依存ではあるが、モデル分離化による編集性<br>
手形状生成VAEをグリッパや他のハンドに変更可である<br>

- ✅ **損失関数**
席取りLossを提案し、VAEの潜在空間を拘束

## プロジェクト全体の流れ

このリポジトリは、**3つのプロジェクト**から構成されています：

### 各プロジェクトの詳細

#### 1️⃣ utils_Pretrained_PointNet/ - パーツ獲得のPointNetの事前学習

**目的**: パーツ分割を行うPointNetの事前学習とScalingNetの事前学習
PointNetのパーツセグメンテーションのNNを学習<br>
物体に対する手のscaleの予測

#### 2️⃣ utils_Pretrained_Hand/ - 手首座標系の手形状生成VAEの事前学習

**目的**: 手形状を生成するVAEを事前学習
手形状は、手首座標系・中指までの骨格長を0.5・親指の向きを[1,0]に平行・[1,0,0]を向くように標準化

#### 3️⃣ utils_Hand_Generation/ - 物体座標系における手の位置・向き生成VAEの学習

**目的**: 事前学習したモデルを転移学習させ、物体座標系に変換するVAEを生成
手の位置・向きは互いに依存していると考えられるため、同一の潜在変数zからそれぞれを生成する

## プロジェクト構成

```
Shape2Gesture_GenerationModel/
│
├── dataset/                                # dataset
├── demo/                                   # demo 
├── Log_tensorboard/                        # logs tensorboard
├── utils_Pretrained_Hand/                  # 【Phase 2】pretrain HandVAE
├── utils_Pretrained_PointNet/              # 【Phase 1】pretrain PointNet, ScalingNet
└── utils_train_Hand_Generation/            # 【Phase 3】pretrain PartsEncoder, train model
```

## 学習フロー
1. **Phase 1 : train PointNet, ScalingNet**
- PointNet: 全体形状 → パーツラベルに属する確率(0:非接触, 1:左手, 2:右手)
- ScalingNet: 全体形状 → 手の大きさ

2. **Phase 2 : train HandVAE**
- HandVAE: 手 → 手

3. **Phase 3 : train PartsEncoder,  PositionVAE**
- PartsEncoder: パーツ → 手 を生成するためのEncoder 
- PositionVAE: パーツ形状・全体形状 → 物体座標系における手の位置・向き


## データセット詳細

### ディレクトリ構成

```
dataset/
├── train/                   # 訓練用データ
│   ├── pts/                 # ポイントクラウド 
│   ├── pts_label/           # セグメンテーションラベル(0:非接触, 1:左手接触, 2:右手接触)
│   └── hands/               # 手ジェスチャー 
│
├── val/                     # 検証用データ
│   ├── pts/
│   ├── pts_label/
│   └── hands/
│
├── search/                  # 検索用データベース（訓練 + 検証）
│   ├── pts/
│   ├── pts_label/
│   └── hands/
│
└──  demo/                    # デモデータ(未知形状)
    └── pts/
```

### 訓練と評価

#### 1️⃣ utils_Pretrained_PointNet/ - パーツ獲得のPointNetの事前学習
```bash
cd utils_Pretrained_PointNet
```

```bash
python pretrained_PointNet \
    --dataset path/to/dataset \
    --batchSize 16 \
    --nepoch 100 \
    --outf path/to/save_model \
```

```bash
python pretrained_ScalingNet \
    --dataset path/to/dataset \
    --batchSize 16 \
    --nepoch 100 \
    --outf path/to/save_model \
```
#### 2️⃣ utils_Pretrained_HandVAE/ - 手形状生成Hand VAEの事前学習

```bash
cd utils_Pretrained_Hand
```

```bash
python pretrained_HandVAE_format_xy.py \
    --dataset path/to/dataset \
    --batchSize 16 \
    --nepoch 100 \
    --outf path/to/save_model \
    --beta 0.1 \
```

```bash
python test_pretrained_HandVAE_format_xy.py \
    --dataset path/to/dataset \
    --idx 0\
    --model path/to/save_model \ #学習したモデルのパス
```

#### 3️⃣ PartsEncoder・PositionVAEの学習
```bash
cd utils_train_PositionVAE
```
##### PartsEncoder
```bash
python train_PartsEncoder.py \
    --dataset path/to/dataset \
    --batchSize 16 \
    --nepoch 100 \
    --outf path/to/save_model \
    --beta 0.1 \
```
##### PositionVAE
```bash
python train_positionVAE_w_batchsampler\
    --dataset path/to/dataset \
    --batchSize 16 \
    --nepoch 100 \
    --outf path/to/save_model \
    --beta 0.1 \
```

```bash
python show_g2p_target.py \
    --dataset path/to/dataset \
    --idx 0\
    --model path/to/save_model \ #学習したモデルのパス
```
[▶ Demo Video (YouTube)](https://youtu.be/axDwgBzbbtc)

[▶ Demo Video (YouTube)](https://youtu.be/EhZxxHoMk1g)


## 必要環境
- Python 3.9
- PyTorch 2.5
- CUDA 11.3, 11.8, 12.1で確認済み
- NumPy, Matplotlib, OpenCV, Pandas, tqdm

## インストール

```bash
git clone https://github.com/yu0225dayo/Shape2Gesture_SearchModel
cd Shape2Gesture_SearchModel

conda env create -f enviroment.yml
conda activate py39_pytorch
```
