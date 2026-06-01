# 项目复现规划：基于 LSTM 的视频无监督表示学习

## 1. 项目概述
本项目旨在复现论文《Unsupervised Learning of Video Representations using LSTMs》中的核心网络架构。项目将使用 **Composite Model（复合模型）** 并在 **Moving MNIST** 数据集上进行无监督训练，实现视频序列的输入重构（Input Reconstruction）与未来预测（Future Prediction）。

## 2. 数据集准备：Moving MNIST
需要编写一个数据生成器或下载标准数据集。根据论文描述，数据需满足以下设定：
* **图像尺寸**：$64 \times 64$ 像素的灰度图。
* **序列长度**：每个视频片段共 20 帧。前 10 帧作为输入，后 10 帧作为未来预测的真实标签（Ground Truth）。
* **内容构成**：每帧包含 2 个手写数字（从 MNIST 训练集中随机抽取），在画面内做匀速直线运动，碰到边缘发生反弹，数字之间可以重叠。

## 3. 网络架构设计：Composite Model
网络使用 Encoder-Decoder 架构，包含一个编码器和两个平行的解码器。

### 3.1 基础单元
* **LSTM 层**：可以使用 1 层或 2 层堆叠的 LSTM。根据论文，单层或双层 LSTM 的隐藏层单元数（Hidden Units）均设置为 **2048**。

### 3.2 Encoder (编码器)
* **输入**：形状为 `(batch_size, 10, 64*64)` 的展平图像序列（10 帧）。
* **行为**：按顺序读取 10 帧输入，提取出最终的隐藏状态（Hidden State）和细胞状态（Cell State）作为整个视频序列的特征表示。

### 3.3 Decoder 1: Input Reconstructor (输入重构器)
* **输入状态**：初始化为 Encoder 的最终状态。
* **输出**：重构过去的 10 帧输入。
* **关键策略（逆序重构）**：目标序列需与输入序列相同，但**顺序颠倒**（即先重构第 10 帧，最后重构第 1 帧），这有助于优化器捕捉短距离相关性。
* **非条件解码（Unconditioned）**：解码过程中不需要将上一帧的输出作为当前步的输入，通常向解码器输入全零向量或特定的启动标记。

### 3.4 Decoder 2: Future Predictor (未来预测器)
* **输入状态**：同样初始化为 Encoder 的最终状态。
* **输出**：预测未来的 10 帧。
* **条件解码（Conditional Decoder 推荐）**：为了获得更好的预测效果，该解码器应当是条件解码器。即：在训练时，将上一帧的真实图像（Ground Truth）作为当前步的输入；在测试（推理）时，将上一步生成的预测图像直接作为当前步的输入。

## 4. 训练策略与超参数
* **输出激活函数**：由于是 Moving MNIST 数据集，解码器的最终输出层应使用 **Logistic (Sigmoid)** 激活函数，将像素值映射到 (0, 1) 区间。
* **损失函数 (Loss Function)**：对于 Moving MNIST，使用**交叉熵损失（Cross Entropy Loss）**。总损失应为重构损失（Reconstruction Loss）与预测损失（Prediction Loss）的加权和（通常权重为 1:1）。
* **优化器 (Optimizer)**：论文明确指出使用 **RMSProp** 优化器，它比带动量的 SGD 收敛得更快。

## 5. Vibe Coding 阶段性实施建议
建议让 AI 助手分步骤完成：
1. **Step 1**：编写 Moving MNIST 数据生成器（利用 PyTorch 的 `Dataset` 和 `DataLoader`），并添加可视化代码以验证生成的序列。
2. **Step 2**：编写基础的 `EncoderLSTM` 和 `DecoderLSTM` 模块，明确张量维度的变化。
3. **Step 3**：组装 `CompositeModel`，实现并行的逆序重构逻辑和基于 Teacher Forcing（针对训练阶段）的条件预测逻辑。
4. **Step 4**：编写训练循环、损失函数计算逻辑，并集成 TensorBoard 或 Wandb 用于记录 Loss 和定期生成预测的 GIF 动图。