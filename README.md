# DSCA-HLAII_Reproduction

本仓库用于复现一个基于 `DSCA-HLAII` 的肽段-HLA II 结合预测流程。当前代码已经包含：

- 训练入口与 YAML 配置支持
- `peptide` ESM embedding 预计算
- `HLA pair` 唯一缓存式 embedding 预计算
- 日志、`metrics.csv`、`last/best checkpoint` 与断点续跑

## 目录结构

```text
.
├── config/
│   └── train.yaml
├── data/
│   ├── hla_dict/
│   ├── medium/
│   └── small/
├── preprocess/
│   ├── precompute_esm.py
│   └── sample_dataset.py
├── src/
│   ├── data_utils.py
│   ├── datasets.py
│   ├── early_stopping.py
│   └── networks.py
├── check.py
├── main.py
└── requirements.txt
```

## 主要文件说明

### 训练与配置

- `main.py`
  - 训练主入口。
  - 负责读取 `config/train.yaml`、构建数据集与 DataLoader、训练/验证循环、日志输出、保存 `last.pt` / `best.pt`、以及断点续跑。

- `config/train.yaml`
  - 默认训练配置。
  - 包含数据路径、embedding 路径、batch size、epoch、optimizer、run 目录、resume 开关等参数。

### 数据与预处理

- `preprocess/precompute_esm.py`
  - 预计算 ESMC embedding。
  - `peptide` embedding 按样本顺序写入单个 `.npy` 文件。
  - `HLA` embedding 按唯一 `alpha-beta pair` 去重保存，并额外生成一个 `names` 索引文件。

- `preprocess/sample_dataset.py`
  - 从原始 `benchmark` 数据构造训练子集。
  - 主要用于按 HLA 等位基因统计后筛选数据。

- `src/data_utils.py`
  - 负责读取 HLA 序列字典。
  - 负责把数据文件与预计算 embedding 组装成训练所需的 `data_list`。
  - 支持按样本 HLA embedding 和“唯一 HLA + names 索引”两种读取模式。

- `src/datasets.py`
  - `HLAIIDataset` 定义。
  - 负责序列截断/补齐、one-hot 编码，以及将 embedding 包装成模型输入。

### 模型与训练辅助

- `src/networks.py`
  - `DSCA_HLAII` 模型定义。
  - 包含残基级特征融合、Transformer/Conv 表征提取、双流交叉注意力和最终预测头。

- `src/early_stopping.py`
  - 早停逻辑。
  - 维护 `best_score`、`counter`、`val_loss_min`，并支持状态保存与恢复。

### 其他脚本

- `check.py`
  - 一个简单的数据统计脚本。
  - 用于查看不同 HLA 等位基因下正负样本数量分布。

- `requirements.txt`
  - 项目依赖列表。

## 数据文件说明

当前仓库中的常见输入/输出文件包括：

- `data/hla_dict/hla_full_seq_dict.txt`
  - HLA 链序列字典。

- `data/small/*.txt` / `data/medium/*.txt`
  - 样本数据文件，通常每行格式为：
  - `peptide alpha_chain beta_chain label`

- `data/pep/*_pep_esm.npy`
  - 按样本顺序保存的 peptide embedding。

- `data/hla/*_hla_unique_esm.npy`
  - 按唯一 HLA pair 保存的 embedding 数组。

- `data/hla/*_hla_unique_names.npy`
  - 与上面的唯一 HLA embedding 行号对应的 `hla_name` 索引文件。

## 基本使用

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 生成 ESM embedding

```bash
python preprocess/precompute_esm.py
```

说明：

- 该脚本依赖环境变量 `HF_TOKEN`
- 如果 `peptide` embedding 已存在，会自动跳过重算
- `HLA` embedding 会按唯一 pair 去重保存

### 3. 开始训练

```bash
python main.py
```

或指定配置文件：

```bash
python main.py --config config/train.yaml
```

训练输出默认保存在：

```text
runs/<run_name>/
├── config.snapshot.yaml
├── metrics.csv
├── train.log
└── checkpoints/
    ├── best.pt
    └── last.pt
```

## 备注

- 目前默认 optimizer 是 `Adadelta`
- 训练采用 mini-batch 方式，`batch_size` 由 `config/train.yaml` 控制
- 如果要断点续跑，请保持同一个 `run.name`，并将 `resume: true` 打开
