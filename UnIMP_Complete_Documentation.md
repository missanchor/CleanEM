# UnIMP 完整文档

## 项目概述

UnIMP (On LLM-Enhanced Mixed-Type Data Imputation with High-Order Message Passing) 是一个用于混合类型数据缺失值插补的深度学习框架，结合了LLM（大语言模型）和超图神经网络。通过训练adapter来连接GNN（图神经网络）和LLM，实现对缺失数据的智能填补。

## 目录结构

```
UnIMP-master/
├── data/                    # 数据集文件夹
│   ├── bike.csv
│   ├── blogger.csv
│   ├── buy.csv
│   ├── chess.csv
│   ├── libras.csv
│   ├── parkinsons.csv
│   ├── phishing.csv
│   ├── restaurant.csv
│   ├── shuttle.csv
│   ├── walmart.csv
│   └── zoo.csv
│
├── models/                  # 模型定义
│   ├── gen_imp.py          # 主生成插补模型 (Gen_IMP)
│   ├── imputaion_model.py  # 插补头模型 (LinearHead, LLMHead)
│   ├── v2e_layer.py       # 节点到超边层
│   ├── e2v_layer.py        # 超边到节点层
│   └── set_transformer.py  # 集合变换器（未使用）
│
├── scripts/                # 运行脚本
│   ├── run_linear_all.sh   # 数值型数据预训练
│   ├── run_LLM_all.sh      # 文本型数据预训练
│   ├── run_finetune.sh     # 微调脚本
│   └── ...                 # 其他实验脚本
│
├── main.py                 # 主入口文件
├── training.py             # 训练模式入口
├── finetune.py             # 微调模式入口
├── testing.py             # 测试模式入口
├── data_loader.py          # 数据加载和处理
├── utils.py                # 工具函数
└── setting.json            # 配置文件
```

## 1. 整体架构

### 1.1 核心组件

UnIMP架构包含两个主要模型：

1. **Gen_IMP（GNN模型）** - 主干网络
   - 位置：`models/gen_imp.py`
   - 作用：学习表格数据的图结构表示
   - 结构：
     ```
     输入: hyperedge, hyper_node, ve_affiliation
       ├── V2E层（节点→超边）: 聚合节点信息到超边
       ├── E2V层（超边→节点）: 聚合超边信息到节点
       └── 循环 gnn_layer_num 次
     输出: hyperedge嵌入, hyper_node嵌入
     ```

2. **LLMHead（Adapter）** - 适配器头部
   - 位置：`models/imputaion_model.py`
   - 作用：将GNN输出与LLM语言能力结合
   - 结构：
     ```
     GNN嵌入 → Linear → 融合模块 ← LLM token嵌入
                     ↓
                  FFN + LayerNorm
                     ↓
                 LM Head → logits
     ```

### 1.2 模型工作流程

```
训练流程：
加载数据 → 构建GNN嵌入 → 结合LLM token嵌入 → 通过Adapter输出 → 计算损失 → 反向传播
```

## 2. 核心文件详解

### 2.1 main.py - 主入口

**功能**: 统一的程序入口，根据模式参数调用不同的模块

**主要组件**:
- `main()`: 参数解析和模式分发
  - `--mode`: 运行模式（training/finetune/testing）
  - `--header_type`: 数据类型（Linear数值型 / LLM文本型）
  - `--data`: 数据集名称
  - `--missing_ratio`: 缺失率
  - `--missing_mechanism`: 缺失机制（MCAR/MAR/MNAR）
  - `--llm_path`: LLM模型路径

**流程**:
```
main.py
  ├── 解析命令行参数
  ├── 加载LLM配置文件
  ├── 设置随机种子和设备
  └── 根据mode调用:
      ├── training.py -> train_model()
      ├── finetune.py -> finetune_model()
      └── testing.py -> test_model()
```

### 2.2 data_loader.py - 数据加载模块

**功能**: 数据预处理、缺失值生成、图结构构建、LLM编码

#### 2.2.1 图结构构建

- `create_edge_node()`: 创建超边节点（行级+列级）
  - 数值型数据：使用one-hot编码
  - 特征维度固定为32
- `create_value_node()`: 创建值节点（单元格值）
- `create_VE_affiliation()`: 创建节点-超边关联矩阵
  - 行级超边：连接行和所有列
  - 列级超边：连接列和所有行

#### 2.2.2 LLM编码（文本型数据）

- `encode()`: 使用LLM对文本进行编码（data_loader.py:126-178）
  ```python
  def encode(texts, tokenizer, model, bs_embedding=256):
      # 1. 分词
      result = tokenizer(texts, padding=True, truncation=True, max_length=256)
      input_ids = result.input_ids
      attention_mask = result.attention_mask
  
      # 2. 生成token embeddings
      hidden_states = model.model(
          input_ids=input_ids,
          attention_mask=attention_mask,
          return_dict=True
      ).last_hidden_state
  
      # 3. 使用最后一个token作为句子嵌入
      sentence_embedding = hidden_states[:, -1]
  
      return sentence_emb, token_emb, labels
  ```
  - 返回: sentence_emb, token_emb, labels

- `create_edge_node_llm()`: 为行/列创建LLM嵌入
- `create_value_node_llm()`: 为单元格值创建LLM嵌入（data_loader.py:189-210）
  - 构建提示文本：`"row i, Given {其他列}, Question: {列名} => {值}"`
  - 完整样本格式：`"row {row_index}, Given {other_cols_str}, Question: {col_name} => {cell_value} <eos>"`
  - 前缀格式（用于生成）：`"row {row_index}, Given {other_cols_str}. Question: {col_name} =>"`

#### 2.2.3 缺失值生成

- `get_data()`: 数值型数据处理
  - 调用 `produce_NA()` 生成缺失值mask
  - 分割训练/测试集
- `get_data_llm()`: 文本型数据处理
  - 使用LLM编码
  - 支持MCAR机制

#### 2.2.4 数据加载主函数

- `load_data()`: （data_loader.py:306-422）
  ```python
  def load_data(args):
      # 加载LLM模型和分词器
      llm_model = AutoModelForCausalLM.from_pretrained(llm_path, device_map="auto")
      tokenizer = AutoTokenizer.from_pretrained(llm_path, device_map="auto")
  
      # 冻结LLM参数
      for param in llm_model.parameters():
          param.requires_grad = False  # 仅训练adapter
  ```
  - 根据 `header_type` 选择处理方式
  - 支持数据分块（chunk）处理
  - 支持嵌入缓存（save_emb/load_emb）
  - 关键步骤：
    1. 加载预训练LLM模型（如Llama2-7B）
    2. 冻结LLM主干参数
    3. 分块处理数据（chunk_size=32 for LLM mode）

**数据流**:
```
CSV文件
  ├── 标准化（数值型）或保持原始（文本型）
  ├── 创建超图结构（超边、值节点、关联矩阵）
  ├── 生成缺失值mask
  ├── 分割训练/测试集
  └── LLM编码（如果是文本型）
```

#### 2.2.5 数据集选择

LLM模式使用的数据集（data_loader.py:321-330）：
- `["drug_test", "guitar_test", "flipkart_test", "SMS_test"]`

### 2.3 models/gen_imp.py - 生成式插补模型

**功能**: 超图神经网络主模型

**核心类**: `Gen_IMP`

- **关键参数**:
  - `hyperedge_dim_hidden`: 超边隐藏维度（默认64）
  - `hyper_node_dim_hidden`: 节点隐藏维度（默认64）
  - `gnn_layer_num`: GNN层数（默认3）

### 2.4 models/imputaion_model.py - 插补头模型

**功能**: 根据嵌入预测缺失值

#### 2.4.1 LinearHead（数值型）

- **输入**: GNN嵌入（行+列嵌入拼接）
- **输出**: 标量值（回归）
- **损失函数**: Huber Loss

#### 2.4.2 LLMHead（文本型）

**类结构**（imputaion_model.py:177-235）:
```python
class LLMHead(nn.Module):
    def __init__(self,
                 input_dims,      # 输入维度（hyperedge_dim_hidden * 2）
                 output_dim,      # 输出维度（vocab_size）
                 hidden_layer_sizes,  # 隐藏层大小
                 hidden_activation='relu',
                 dropout=0.2,
                 relation_type="cross_attn"):  # 融合策略
```

**核心组件**：
1. **lm_head** - 预训练语言模型头部
   - 初始化时加载预训练LLM的lm_head权重（training.py:203-207）
   - 用于将隐藏状态映射到词汇表空间

2. **lin_gnn** - GNN特征线性变换
   - 将GNN输出投影到适配维度

3. **fuse_model** - 特征融合模块
   - 支持多种融合策略
   - 默认使用"cross_attn"（交叉注意力机制）

4. **FFN** - 前馈网络
   - 2层MLP，增强模型表达能力

**输入**: 
  - GNN嵌入（行+列嵌入）
  - LLM token嵌入

**输出**: 词汇表大小的logits（分类）

**损失函数**: Cross Entropy Loss

**融合方式** (`relation_type`，imputaion_model.py:195-206）:
  - `attention`: AttentionFusion - 注意力融合
  - `weightedsum`: WeightedSumFusion - 加权求和
  - `gated`: GatedFusion - 门控机制
  - `cross_attn`: CrossAttentionFusion - 交叉注意力（默认）
  - `film`: FiLMFusion - FiLM条件调制

### 2.5 training.py - 训练模块

**功能**: 模型预训练

**核心组件**:
- `HyperBatch`: 批处理数据结构
  - 合并多个chunk的数据
  - 处理节点/超边索引对齐
- `train_model()`: 主训练函数

**训练流程**（training.py:251-295）:
```python
# 核心训练循环
for epoch in range(args.epochs):
    for ids in train_loader:
        # 1. 构建batch
        batch = HyperBatch.from_data_list(...)

        # 2. 生成缺失掩码
        known_mask = produce_NA(..., p_miss=1-args.known, mecha="Random")

        # 3. 前向传播
        embedding, hyper_node = model(hyperedge, known_hyper_node, known_ve_affiliation)

        # 4. Adapter预测
        pred = impute_model([embedding[...], embedding[...]], train_tokens_emb)

        # 5. 计算损失（交叉熵）
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(pred_train.view(-1, pred_train.size(-1)), label_train.view(-1))

        # 6. 反向传播
        loss.backward()
        optimizer.step()
```

详细步骤：
1. 加载数据
2. 初始化模型（Gen_IMP + ImputationHead）
3. 创建优化器（AdamW）
4. 训练循环:
   - 随机mask部分已知值（1-args.known）
   - 前向传播（GNN → ImputationHead）
   - 计算损失（Huber/CrossEntropy）
   - 反向传播
   - 定期评估（每eval_epoch_gap轮）
     - 生成预测
     - 计算指标（RMSE/MAE 或 BLEU/ROUGE）
     - 保存模型

**特殊处理**:
- 数值型: 使用已知mask的训练策略
- 文本型: 使用LLM token嵌入进行自回归预测

**关键参数**:
- `known=0.6` - 已知数据比例（40%缺失）
- `missing_ratio=0.2` - 缺失率（20%）
- `missing_mechanism="MCAR"` - 缺失机制（完全随机缺失）

### 2.6 finetune.py - 微调模块

**功能**: 在预训练模型基础上微调

**与training的区别**:
- **必须加载预训练模型** (`--load_model_name`)
- **渐进式mask策略**（finetune.py:260）:
  ```python
  p_miss_ratio = np.linspace(0.65, 0.35, args.epochs)
  ```
  - 从65%缺失率逐步降低到35%
  - 实现课程学习（curriculum learning）
- **加载预训练模型**（finetune.py:202-225）
  - 首先加载预训练权重，然后继续训练
- **使用Huber损失**（finetune.py:298）
  - 对数值型数据更鲁棒
- **适用于**: 针对特定数据集的进一步优化

### 2.7 testing.py - 测试模块

**功能**: 评估已训练模型

**流程**:
```
1. 加载预训练模型
2. 模型设置为eval模式
3. 对测试集进行预测
4. 计算评估指标
   ├── 数值型: RMSE, MAE
   └── 文本型: BLEU, ROUGE-1/L/Lsum/W/S, Jaccard, Levenshtein, Cosine等
```

**文本生成**（training.py:89-138）:
```python
def generate_impute(args, embedding, impute_model, test_ve_affiliation,
                   lm_model, tokenizer, x_text_test, max_new_tokens=16):
    # 1. 编码查询文本
    inputs = tokenizer(x_text_test, padding=True, truncation=True, return_tensors="pt")

    # 2. 循环生成token
    for _ in range(max_new_tokens):
        # 获取LLM隐藏状态
        outputs = lm_model.model(input_ids=generated, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        # 使用adapter预测下一个token
        logits = impute_model([embedding[...], embedding[...]], hidden_states)

        # 采样或贪心选择
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)

        # 更新生成序列
        generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

    # 3. 解码生成文本
    return [tokenizer.decode(gen, skip_special_tokens=True) for gen in generated]
```

- `generate_impute()`: 自回归生成缺失值
  - 使用LLM生成token序列
  - 最大生成长度: max_new_tokens

### 2.8 utils.py - 工具函数

**功能**: 辅助功能实现

#### 2.8.1 缺失值机制 (`produce_NA`, utils.py:322-373)

支持三种缺失机制：
- `MCAR`: 完全随机缺失（Missing Completely At Random）
- `MAR`: 随机缺失（Missing At Random，基于逻辑回归）
- `MNAR`: 非随机缺失（Missing Not At Random）
  - `logistic`: 逻辑回归mask
  - `quantile`: 分位数mask
  - `selfmasked`: 自mask逻辑回归

#### 2.8.2 LLM生成评估 (`compute_LLM_generation_metrics`, utils.py:481-574)

评估指标:
- **BLEU Score** - 机器翻译评估
- **ROUGE系列**: ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum, ROUGE-W, ROUGE-S - 文本摘要评估
- **Jaccard相似度** - 集合重叠度
- **Levenshtein距离** - 编辑距离
- **Cosine相似度**（多种变体） - 语义相似度
  - 字符级
  - TF（词频）
  - TF-IDF
  - Word Embeddings（BERT）

#### 2.8.3 其他工具

- `get_activation()`: 激活函数选择
- `get_main_device()`: 获取模型主设备

## 3. 数据流图

### 3.1 训练流程（数值型）

```
CSV数据
  ↓
数据标准化 + 创建超图结构
  ↓
生成缺失值mask (MCAR/MAR/MNAR)
  ↓
分割训练/测试集
  ↓
GNN模型（Gen_IMP）
  ↓ 节点/超边嵌入
LinearHead
  ↓ 预测值
Huber Loss → 反向传播
```

### 3.2 训练流程（文本型）

```
CSV数据（文本）
  ↓
构建提示文本
  ↓
LLM编码 → sentence_emb, token_emb
  ↓
生成缺失值mask
  ↓
GNN模型（Gen_IMP）
  ↓ 节点/超边嵌入
LLMHead（融合GNN嵌入和token嵌入）
  ↓ 词汇表logits
CrossEntropy Loss → 反向传播
```

### 3.3 测试流程（文本型）

```
测试数据
  ↓
GNN生成嵌入
  ↓
generate_impute() 自回归生成
  ├── LLM生成token
  ├── LLMHead预测下一个token
  └── 重复直到生成完整值
  ↓
解码 → 文本预测值
  ↓
计算评估指标（BLEU, ROUGE等）
```

### 3.4 完整训练流程总结

```
1. 加载预训练LLM（如Llama2-7B）
   ↓
2. 冻结LLM主干参数
   ↓
3. 初始化Adapter（LLMHead）
   ↓
4. 加载LLM的lm_head权重到Adapter
   ↓
5. 构建表格数据的图表示
   ↓
6. 编码cell文本为token embeddings
   ↓
7. 模拟缺失数据（MCAR/MAR/MNAR）
   ↓
8. 训练循环：
   - GNN前向传播 → 生成嵌入
   - Adapter融合GNN嵌入和LLM tokens
   - 计算交叉熵损失
   - 反向传播更新参数
   ↓
9. 保存模型检查点
   ↓
10. 推理：使用adapter进行自回归生成
```

## 4. 超图结构

### 4.1 节点类型

1. **超边节点（hyperedge）**:
   - 行级超边: 代表数据行
   - 列级超边: 代表数据列

2. **值节点（hyper_node）**:
   - 单元格值（数值或嵌入向量）

### 4.2 关联关系 (VE affiliation)

- 行i ↔ 列j ↔ 单元格(i,j)的值
- 形成二部图结构

### 4.3 消息传递

```
V2E层: 值节点 → 超边节点
  - 聚合同一行/列的所有单元格值
E2V层: 超边节点 → 值节点
  - 聚合同一行/列的所有信息
```

### 4.4 表格到图的转换

1. **创建超图节点**（data_loader.py:180-187）
   - 行级别节点：使用LLM编码描述性文本
   - 列级别节点：使用LLM编码列名和类型信息

2. **创建值节点**（data_loader.py:213-252）
   - 每个cell转化为文本描述
   - 通过LLM获取token embeddings

3. **构建超图边**（data_loader.py:75-84）
   - 行-值边和列-值边
   - 形成双向超图结构

## 5. 模型参数分析

### 5.1 可训练参数（training.py:215-226）

```python
trainable_parameters = list(model.parameters()) \
                     + list(impute_model.parameters())

# 统计参数数量
trainable_params = [p for p in model.parameters() if p.requires_grad]
trainable_params_impute = [p for p in impute_model.parameters() if p.requires_grad]

print('total trainable params in GNN model:', sum(p.numel() for p in trainable_params))
print('total trainable params in impute model:', sum(p.numel() for p in trainable_params_impute))
```

**参数分布**:
- **GNN模型参数**：Gen_IMP所有层
- **Adapter参数**：LLMHead中的lin_gnn、fuse_model、ffn和lm_head
- **LLM主干参数**：冻结状态，不参与训练

### 5.2 LLM权重初始化（training.py:203-207）

```python
# 从预训练LLM加载lm_head权重
impute_model.lm_head.weight.data = lm_model.lm_head.weight.data.clone()
if lm_model.lm_head.bias is not None:
    impute_model.lm_head.bias.data = lm_model.lm_head.bias.data.clone()
else:
    impute_model.lm_head.bias.data = torch.zeros_like(impute_model.lm_head.bias.data)
```

## 6. 损失函数

### 6.1 训练损失（training.py:290-291）

**LLM模式使用交叉熵损失**:
```python
loss_fct = nn.CrossEntropyLoss()
loss = loss_fct(pred_train.view(-1, pred_train.size(-1)), label_train.view(-1))
```

**数值型使用Huber损失**:
```python
loss_fct = nn.HuberLoss()
loss = loss_fct(pred_train, label_train)
```

### 6.2 评估指标

训练完成后评估生成质量（详见2.8.2节）：
- **BLEU Score** - 机器翻译评估
- **ROUGE Scores**（1, 2, L, Lsum, W, S） - 文本摘要评估
- **Jaccard相似度** - 集合重叠度
- **Levenshtein距离** - 编辑距离
- **Cosine相似度**（多种变体） - 语义相似度

## 7. 配置参数

### 7.1 模型参数

- `hyperedge_dim_hidden`: 64 (超边隐藏维度)
- `hyper_node_dim_hidden`: 64 (节点隐藏维度)
- `gnn_layer_num`: 3 (GNN层数)
- `imputer_layer_num`: 1 (插补层数)

### 7.2 训练参数（main.py:16-64）

```python
# 训练参数
epochs = 4000
lr = 0.001
weight_decay = 0.0
dropout = 0.0

# 缺失数据参数
known = 0.6  # 已知数据比例（40%缺失）
missing_ratio = 0.2  # 缺失率（20%）
missing_mechanism = 'MCAR'  # 缺失机制

# 数据处理参数
chunk_size = 32  # LLM模式（数值型为500）
chunk_batch = 32
bs_embedding = 32
```

### 7.3 数据参数

- `missing_ratio`: 0.2 (缺失率)
- `missing_mechanism`: MCAR/MAR/MNAR
- `header_type`: Linear/LLM

### 7.4 保存与加载

**保存模型**（training.py:301-302）:
```python
torch.save(model.state_dict(), f"./saved_models/llm_gnn_model_{args.header_type}_Epoch{epoch}_{args.save_name}.pth")
torch.save(impute_model.state_dict(), f"./saved_models/llm_impute_model_{args.header_type}_Epoch{epoch}_{args.save_name}.pth")
```

**加载模型**（training.py:211-213）:
```python
model.load_state_dict(torch.load(f"./saved_models/llm_gnn_model_{args.load_model_name}.pth"))
impute_model.load_state_dict(torch.load(f"./saved_models/llm_impute_model_{args.load_model_name}.pth"))
```

## 8. 关键技术点

1. **混合类型数据处理**:
   - 数值型: 线性头 + 回归损失（Huber Loss）
   - 文本型: LLM头 + 分类损失（Cross Entropy Loss）

2. **超图神经网络**:
   - 高阶消息传递（节点↔超边）
   - 捕获行列关联关系

3. **LLM集成**:
   - 文本数据编码
   - 自回归生成缺失值
   - 多种融合策略（5种）

4. **缺失值机制模拟**:
   - MCAR/MAR/MNAR三种机制
   - 逻辑回归/分位数等生成方式

5. **课程学习**:
   - 微调时渐进式增加缺失率（从65%到35%）

6. **Adapter架构**:
   - 仅训练小部分参数（adapter），保持LLM冻结
   - 参数效率高，减少计算资源需求

## 9. 创新点

1. **双模块设计**：GNN学习结构信息，LLM提供语言生成能力
2. **Adapter架构**：仅训练小部分参数（adapter），保持LLM冻结
3. **多策略融合**：支持5种不同的GNN-LLM融合策略（attention, weightedsum, gated, cross_attn, film）
4. **超图建模**：将表格数据建模为超图，捕获行和列关系
5. **渐进式微调**：从高缺失率逐渐降低到低缺失率

## 10. 优势

1. **参数效率**：仅训练adapter，减少计算资源需求
2. **迁移能力**：预训练LLM的知识可用于数据填补
3. **灵活性**：可适配不同类型的LLM
4. **鲁棒性**：支持多种缺失数据机制（MCAR/MAR/MNAR）
5. **混合类型支持**：同时处理数值型和文本型数据

## 11. 潜在改进方向

1. **融合策略优化**：探索更有效的GNN-LLM融合方法
2. **长序列建模**：改进对长表格的处理能力
3. **多任务学习**：同时处理多个填补任务
4. **知识蒸馏**：将LLM知识更高效地迁移到adapter

## 12. 依赖关系图

```
main.py
  ├── training.py
  │     ├── data_loader.py
  │     ├── models/gen_imp.py
  │     └── models/imputaion_model.py
  ├── finetune.py (同training依赖)
  └── testing.py (同training依赖)
        │
        └── utils.py (所有模块共享)
```

## 13. 实验脚本说明

- `run_linear_all.sh`: 批量训练数值型数据集
- `run_LLM_all.sh`: 批量训练文本型数据集
- `run_finetune.sh`: 微调指定数据集
- `run_UnIMP.sh`: 标准训练流程
- `run_MissingRate.sh`: 不同缺失率实验
- `run_batchsize.sh`: 不同batch size实验

## 14. 输出文件

- `saved_models/`: 模型权重
  - `llm_gnn_model_{type}_Epoch{epoch}_{name}.pth`
  - `llm_impute_model_{type}_Epoch{epoch}_{name}.pth`
- `figures/`: 训练曲线图（loss, RMSE, MAE）
- `prompt_embedding/`: 缓存LLM嵌入（如果启用save_emb）
- `logs/`: 运行日志

## 15. 关键文件路径总结

```
.
├── main.py              # 主入口，参数配置
├── training.py          # 训练脚本
├── finetune.py          # 微调脚本
├── testing.py           # 测试脚本
├── data_loader.py       # 数据加载和预处理
├── models/
│   ├── imputaion_model.py  # LLMHead adapter实现
│   ├── gen_imp.py         # Gen_IMP GNN模型
│   ├── e2v_layer.py       # Edge-to-Value层
│   └── v2e_layer.py       # Value-to-Edge层
└── utils.py             # 工具函数（缺失数据、评估指标等）
```

