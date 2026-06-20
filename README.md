# bert_lora

---

# 中文电商评论情感二分类系统（BERT + LoRA）

基于 BERT 与 LoRA 参数高效微调技术的中文电商评论情感分析系统。针对真实场景下的用户评论数据，实现端到端的情感极性（正面/负面）自动分类。

**项目特点**：
- 参数高效微调：LoRA 适配器仅训练 **0.45%** 的模型参数
- 训练稳定性优化：Warmup + 梯度裁剪 + 梯度累积
- 实验可复现：自动保存最优模型、超参数日志、错误样本分析报告


## 项目结构

```
├── train_bert_finally.py              # 训练脚本（SentimentTrainer 类）
├── test_bert_finally.py               # 测试脚本（ModelTester 类）
├── shop_2class.csv       # 训练数据集
├── test_reviews.csv      # 测试数据集
├── shop_sentiment_model/ # 训练好的模型
└── test_results.json     # 测试结果报告
```


## 环境依赖

```
Python 3.9+
torch
transformers
datasets
peft
scikit-learn
pandas
numpy
```

安装命令：

```bash
pip install torch transformers datasets peft scikit-learn pandas numpy
```


## 数据集格式

CSV 文件需包含两列：

| 列名 | 说明 |
|------|------|
| `review` | 用户评论文本（中文） |
| `label` | 情感标签，`1` 表示正面，`0` 表示负面 |

**自动容错**：如果 CSV 列名不匹配（如无表头或列名不同），代码会自动修正为 `['label', 'review']`。

示例数据：

| review | label |
|--------|-------|
| 快递速度很快，非常满意 | 1 |
| 质量太差了，不会再买 | 0 |
| 还行吧，凑合用 | 1 |


## 快速开始

### 1. 训练模型

将训练数据命名为 `shop_2class.csv` 放在项目根目录，然后运行：

```bash
python train.py
```

训练过程会自动：
- 清洗数据（空值剔除、字符串去空格、类型强制转换）
- 按 8:2 分层采样划分训练集/验证集
- 加载 BERT-base-Chinese + LoRA 微调
- 每轮评估并自动保存验证集最优模型
- 记录超参数至 JSON 日志

### 2. 测试模型

将测试数据命名为 `test_reviews.csv` 放在项目根目录，然后运行：

```bash
python test.py
```

测试过程会自动：
- 加载训练好的模型
- 逐批次预测并计算准确率和 F1
- 输出预测错误的样本（含文本内容）
- 生成 `test_results.json` 详细结果文件


## LoRA 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `r` | 32 | 低秩矩阵维度 |
| `lora_alpha` | 64 | LoRA 缩放系数 |
| `target_modules` | query, key, value, output.dense | 适配器作用的注意力子层 |
| `lora_dropout` | 0.05 | 正则化 dropout |
| `modules_to_save` | classifier | 分类头全量更新 |
| **可训练参数占比** | **0.45%** | 显著降低显存占用 |


## 训练策略

| 策略 | 参数 | 目的 |
|------|------|------|
| Warmup | `warmup_ratio=0.1` | 前 10% 步数线性增加学习率，稳定初始训练 |
| 梯度裁剪 | `max_grad_norm=1.0` | 防止梯度爆炸 |
| 梯度累积 | `accumulation_steps=2` | 用 2 步小 batch 模拟大 batch 效果，缓解显存不足 |
| 混合精度 | `fp16=True` | 加速训练、减少显存占用（仅 GPU 可用时开启） |
| 早停保存 | `load_best_model_at_end=True` | 训练结束后加载验证集最优模型 |


## 实验结果

| 指标 | 结果 |
|------|------|
| 测试准确率 | **91%** |
| 加权 F1 | **0.91** |

> 任务为二分类（正面/负面），随机猜测基线为 50%。

测试完成后，`test_results.json` 会保存每条样本的预测结果和是否正确，便于错误分析。


## 错误诊断示例

测试脚本会输出预测错误的样本，便于定位模型边界。常见错误类型包括：

- **短文本**：如用户只写“还行”，信息量不足，模型难以判断
- **含否定词的复杂句式**：如“没有想象中那么好”，模型易误判为正面

这些错误样本可直接用于反哺数据清洗与模型优化。


## 实验追踪

训练过程自动在模型目录保存 `training_args.json`，记录：

- 超参数（学习率、batch_size、epochs 等）
- 训练策略（warmup_ratio、gradient_accumulation_steps 等）
- 评估指标（准确率、F1）

便于不同实验版本间的对比与复现。


## 未来优化方向

- 针对短文本进行数据增强（如回译、同义词替换）
- 引入情感词典辅助判断否定句式
- 尝试其他参数高效微调方法（Adapter、Prefix-Tuning）


## License

MIT


## 作者

倪皓轩 - [GitHub](https://github.com/Haozige2003)

---
