# DKF_weather — 基于深度卡尔曼滤波的降水预测

本项目使用 **Deep Kalman Filter (DKF)** 进行短时降水预测，并以 **ConvLSTM**（编码-预测架构）作为对比基线。先在 Moving MNIST 数据集上验证模型正确性，再迁移到真实气象数据（GPM IMERG 卫星降水 + GEBCO 地形）。

> ⚠️ 这是一个课程作业项目。受限于数据量（750 个训练样本）和模型规模，降水预测效果尚不理想，但 DKF 在几乎所有指标上均系统性优于 ConvLSTM。在有充足数据和更大网络的条件下，DKF 应当有更好的表现。

---

## 项目结构

```
DKF_weather/
├── DKF/                          # DKF on Moving MNIST
│   ├── modules_dkf.py            # DKF 模型定义
│   ├── train.py                  # 训练脚本
│   ├── eval.py / eval_1.py       # 评估和可视化
│   └── dataset.py                # MovingMNIST 数据加载
├── ConvLSTM/                     # ConvLSTM on Moving MNIST
│   ├── modules.py                # ConvLSTM + PatchEmbedding
│   ├── train.py                  # 训练脚本
│   ├── eval.py                   # 评估脚本
│   └── dataset.py                # MovingMNIST 数据加载（含 Patchify）
├── DKF_pre/                      # DKF on Precipitation
│   ├── modules.py                # DKF 降水版本（Softplus 输出）
│   ├── train.py                  # 训练脚本（多种降水损失函数）
│   ├── eval.py / eval_train.py   # 评估和可视化
│   └── dataset.py                # 气象数据加载（降水 + 地形）
├── ConvLSTM_pre/                 # ConvLSTM on Precipitation
│   ├── modules.py                # PrecipConvLSTM（2 通道输入）
│   ├── train.py                  # 训练脚本
│   ├── eval.py                   # 评估脚本（含散点图）
│   └── dataset.py                # 气象数据加载（返回 past/future）
├── data/
│   ├── MovingMNIST/              # Moving MNIST 数据集
│   │   └── mnist_test_seq.npy
│   └── data_0511/               # 降水 + 地形数据
│       ├── nc_NZL/               # GPM IMERG 降水 NetCDF 文件
│       ├── gebco0_1_land_only.nc # 地形数据（仅陆地）
│       └── readme.txt            # 数据说明
├── comparison_results/           # 评估和对比结果（gitignored）
│   ├── moving_mnist/             # Moving MNIST DKF vs ConvLSTM 对比
│   ├── dkf_pre_precip/           # DKF 降水评估结果
│   └── precip_comparison/        # 降水 DKF vs ConvLSTM 全面对比
├── compare_moving_mnist.py       # Moving MNIST 对比脚本
├── compare_dkf_convlstm_precip.py # 降水对比脚本（双栏论文图表）
├── evaluate_dkf_pre_precip.py    # DKF 降水独立评估脚本
├── requirements.txt
├── .gitignore
└── README.md
```

## 模型架构

| 组件 | DKF | ConvLSTM |
|---|---|---|
| 输入处理 | ObservationEncoder (CNN) | PatchEmbedding (4×4 patches) |
| 时序建模 | InferenceNetwork + TransitionNetwork (ConvLSTM-based) | Multi-layer ConvLSTM (3 层) |
| 输出层 | ObservationDecoder → Softplus | PatchRecovery → Softplus |
| 训练方式 | KL Divergence + Reconstruction Loss，Beta warmup，Scheduled Sampling | BCEWithLogitsLoss (MNIST) / Asymmetric Heavy-Rain Loss (降水) |
| 推理方式 | 概率推断（Ensemble 预测） | 确定性前向传播 |

### DKF 原理

Deep Kalman Filter 将卡尔曼滤波与深度学习结合：
1. **Encoder**：将观测映射到潜在空间
2. **Inference Network**：给定过去观测，推断潜在状态的后验分布 `q(z_t | x_{1:t})`
3. **Transition Network**：在潜在空间中建模状态转移 `p(z_t | z_{t-1})`
4. **Decoder**：从潜在状态重建观测 `p(x_t | z_t)`
5. 训练目标 = **重建损失 + β·KL 散度**（VAE 风格），使潜在状态同时具有预测能力和生成能力

## 数据集

### 1. Moving MNIST

- 来源：`mnist_test_seq.npy`（可从 [Google Drive](https://www.cs.toronto.edu/~nitish/unsupervised_video/) 下载）
- 形状：`(10000, 20, 1, 64, 64)` — 10000 个序列，每序列 20 帧，单通道 64×64
- 像素值：`[0, 255]`，加载时归一化到 `[0, 1]`
- 划分：80% 训练 / 20% 测试（seed=42）
- **放置路径**：`data/MovingMNIST/mnist_test_seq.npy`

### 2. 气象降水数据（GPM IMERG + GEBCO 地形）

| 数据 | 来源 | 变量 | 时空分辨率 |
|---|---|---|---|
| GPM IMERG 降水 | NASA GES DISC | `GPM_3IMERGHH_07_precipitation` | 0.1° × 30 min |
| GEBCO 地形 | GEBCO | `elevation` | 0.1°（最近邻升尺度） |

- **区域**：新西兰周边（-32.25°S ~ -47.95°S, 163.1°E ~ 179.6°E），网格 158×165
- **时间**：2020 年 7 月 1 日–23 日（南半球雨季），每 30 分钟一帧，共约 1060 帧
- **地形**：陆地保留原始海拔，海洋赋值为 0（`gebco0_1_land_only.nc`）
- **预处理**：降水做 `log1p` 变换以压缩偏态分布；地形做 Min-Max 归一化
- **划分**：train `[:750]`，val `[750:900]`，test `[900:]`
- **放置路径**：
  ```
  data/data_0511/nc_NZL/*.nc                  # 降水 NetCDF 文件（约 46 个）
  data/data_0511/gebco0_1_land_only.nc         # 地形数据
  ```

> 降水数据可从 [NASA GPM](https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGHH_07/summary) 下载，地形数据可从 [GEBCO](https://www.gebco.net/data_and_products/gridded_bathymetry_data/) 下载。

## 环境配置

```bash
pip install torch numpy matplotlib xarray netCDF4 tqdm tensorboard
```

CUDA 可选，CPU 也能训练（速度较慢）。

## 训练

### Moving MNIST 验证

```bash
# 训练 DKF on Moving MNIST (~350 epochs, 约 3-4 小时 GPU)
cd DKF
python train.py

# 训练 ConvLSTM on Moving MNIST (~100 epochs, 约 1 小时 GPU)
cd ConvLSTM
python train.py
```

### 降水预测

```bash
# 训练 DKF on Precipitation (~100 epochs)
cd DKF_pre
python train.py

# 训练 ConvLSTM on Precipitation (~100 epochs)
cd ConvLSTM_pre
python train.py
```

#### ConvLSTM_pre 训练参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--precip_data_path` | `../data/data_0511/nc_NZL/*.nc` | 降水数据路径 |
| `--land_data_path` | `../data/data_0511/gebco0_1_land_only.nc` | 地形数据路径 |
| `--batch_size` | 32 | 批次大小 |
| `--epochs` | 100 | 训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--n_past` | 10 | 输入帧数 |
| `--n_future` | 10 | 预测帧数 |
| `--hidden_dim_list` | 128 64 64 | ConvLSTM 隐层通道数 |
| `--patch_size` | 4 | Patch 大小 |
| `--grad_clip` | 5.0 | 梯度裁剪 |
| `--save_dir` | `./checkpoints` | 模型保存目录 |
| `--resume_checkpoint` | None | 断点续训路径 |

### 断点续训

```bash
python train.py --resume_checkpoint ./checkpoints/last_model.pth
```

## 评估与对比

### 单模型评估

```bash
# DKF 降水评估
python evaluate_dkf_pre_precip.py

# ConvLSTM 降水评估
cd ConvLSTM_pre
python eval.py --checkpoint ./checkpoints/best_model.pth
```

### 全面对比

```bash
# Moving MNIST: DKF vs ConvLSTM
python compare_moving_mnist.py

# 降水: DKF vs ConvLSTM（生成论文图表）
python compare_dkf_convlstm_precip.py
```

对比脚本会输出：
- 连续误差指标（MSE, RMSE, MAE, Bias, Pearson r）
- 逐预测步误差分解
- 分类降水评分（POD, FAR, CSI, F1 @ 0.1/2/5/10/30 mm/h 阈值）
- 降水预报可视化对比图
- 散点密度图、柱状图

结果保存在 `comparison_results/` 目录下。

## 主要结果

### Moving MNIST（验证阶段）

| 指标 | ConvLSTM | DKF | 胜出 |
|---|---|---|---|
| MSE | 0.0219 | 0.0223 | ConvLSTM |
| MAE | 0.0469 | **0.0341** | DKF |
| SSIM | 0.8053 | **0.8629** | DKF |

### 降水预测（核心任务）

**连续误差**（DKF 全面领先）：

| 指标 | ConvLSTM | DKF | 改善 |
|---|---|---|---|
| MSE | 0.6144 | **0.4247** | −30.9% |
| MAE | 0.3181 | **0.1879** | −40.9% |
| Bias | 0.1764 | **0.0168** | — |
| Pearson r | 0.4318 | **0.5116** | +18.5% |

**关键发现**：
- DKF 的 Bias 仅 0.017 mm/h（接近零偏），而 ConvLSTM 系统性高估 2.15 倍
- ConvLSTM 降水检出率（POD）更高，但空报率（FAR）高达 72.7%——"什么都报，什么都报不准"
- DKF 在所有降水阈值上的 CSI 和 F1 均优于 ConvLSTM
- 两模型均无法预测 ≥30 mm/h 的极端降水

> 详细分析见 `comparison_results/precip_comparison/metrics_summary.md`

## 局限性 & 改进方向

1. **数据量不足**：750 个训练样本远不足以训练可靠的降水预测模型。DKF 作为概率模型通常需要更多数据来学习稳定的潜在动态。
2. **网络规模有限**：受限于 GPU 显存和训练时间，ConvLSTM 仅 3 层，DKF 的编码器/解码器也较轻量。
3. **极端降水**：`log1p` 变换压缩了强降水信号，模型对 ≥10 mm/h 的降水预测能力很弱。
4. **改进方向**：
   - 增大训练数据（数月甚至数年的降水记录）
   - 增加模型容量（更深、更宽的编码器/解码器）
   - 引入对强降水加权的损失函数
   - 尝试在原始空间（不做 log 变换）建模，或使用更合适的分布假设（如 Gamma 分布）

## 依赖

- Python ≥ 3.8
- PyTorch ≥ 1.10
- NumPy, Matplotlib, xarray, netCDF4
- tqdm, TensorBoard
