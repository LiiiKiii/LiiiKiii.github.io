# AI-Pedia 评估系统 (Evaluation System)

## 📋 概述

本评估系统实现了论文中描述的"同一基础模型对比"评估方法：

1. **受限基线 (Restricted Baseline)**: 使用简单关键词匹配，无高级处理
2. **完整 Pipeline (Full Pipeline)**: TF-IDF + MMR + AI 过滤 + CBF 排名

这种设计确保任何观察到的改进都可以归因于 AI-Pedia pipeline 组件，而非底层模型差异。

## 🎯 评估方法

### 核心策略

```
相同基础模型 (GPT-3.5-turbo)
    ↓
  ┌──────────────┬──────────────┐
  ↓              ↓              ↓
受限基线      完整Pipeline    指标对比
(简单匹配)    (全部处理)     (量化提升)
```

### 研究问题

| 问题 | 指标 | 说明 |
|------|------|------|
| RQ1 | Coverage, Diversity, AI Relevance | TF-IDF + MMR 能否产生稳定主题信号？ |
| RQ2 | AI Relevance, Noise Reduction, Diversity | 多源检索 + AI 过滤能否改善资源质量？ |
| RQ3 | Novelty, Authority Score, Valid URLs | CBF 排名能否提升资源质量？ |

## 📁 目录结构

```
test/evaluation_pipeline/
├── __init__.py          # 包初始化
├── config.py            # 配置管理（环境变量）
├── metrics.py           # 评估指标计算
├── evaluator.py         # 主评估脚本
└── results/             # 评估结果输出
```

## 🚀 快速开始

### 1. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env，设置测试路径和 API Key
nano .env
```

### 2. 准备测试语料

创建测试语料目录：

```bash
mkdir -p data/test_corpus
# 放入 10+ 个 AI/ML 相关的 .txt 文件
```

### 3. 运行评估

```bash
cd /path/to/AI-Pedia/Project

# 基础运行
python test/evaluation_pipeline/evaluator.py

# 指定语料路径
python test/evaluation_pipeline/evaluator.py --corpus data/test_corpus

# 指定输出目录
python test/evaluation_pipeline/evaluator.py --corpus data/test_corpus --output test/evaluation_pipeline/results
```

## 📊 评估指标

### 主要指标

#### 1. AI Relevance（AI 相关性）
**定义**: 推荐资源中与 AI/ML 相关的比例

**计算**:
```python
relevance = (AI相关资源数 / 总资源数) × 100%
```

#### 2. Noise Reduction（噪声减少率）
**定义**: AI 过滤器成功去除的无关内容比例

**计算**:
```python
noise_reduction = ((初始资源数 - 过滤后资源数) / 初始资源数) × 100%
```

#### 3. Cross-Platform Diversity（跨平台多样性）
**定义**: 不同来源平台类型的数量

**平台类型**:
- Text: Wikipedia, arXiv, Google Scholar
- Video: YouTube, Bilibili
- Code: GitHub, GitLab, Colab

#### 4. Authority Score（权威性分数）
**定义**: 来自可信来源的资源比例

**可信来源**: arXiv, Wikipedia, GitHub, PyTorch.org, TensorFlow.org 等

#### 5. Novelty（新颖性）
**定义**: 推荐资源中超出原始语料的新概念比例

#### 6. Valid URLs（有效链接）
**定义**: 可访问的资源链接比例

### 消融研究指标

| 配置 | AI Relevance | 跨平台多样性 |
|------|-------------|-------------|
| 完整 Pipeline | 91% | 4.7 |
| 无 MMR | 89% | 3.8 |
| 无 AI Filter | 41% | 4.6 |
| 无 CBF Ranking | 52% | 3.1 |
| 无规范化 | 88% | 4.5 |

## 🔧 配置说明

### config.py

```python
class EvalConfig:
    # OpenAI API Key（从环境变量读取）
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    # 默认基础模型
    DEFAULT_FOUNDATION_MODEL = "gpt-3.5-turbo"

    # AI 领域关键词列表
    AI_RELEVANCE_KEYWORDS = [
        "machine learning", "deep learning", "neural network", ...
    ]
```

### 环境变量

在 `.env` 中设置：

```bash
OPENAI_API_KEY=your_api_key_here
TEST_CORPUS_PATH=./data/test_corpus
EVAL_OUTPUT_DIR=./test/evaluation_pipeline/results
```

## 📈 示例输出

### 控制台输出

```
============================================================
AI-Pedia Evaluation: Restricted vs Full Pipeline
============================================================
📂 Loaded 50 documents from corpus

============================================================
Running RESTRICTED Baseline (Naive Search)...
============================================================
📌 Extracted keywords: ['learning', 'network', 'data', 'model', ...]
📥 Found 1247 resources

📊 Restricted Baseline Results:
  total_resources: 1247
  ai_relevance: 32.0
  cross_platform_diversity: 3
  authority_score: 28.0
  url_validation: {'valid': 1106, 'invalid': 141, 'valid_percentage': 88.7}

============================================================
Running FULL AI-Pedia Pipeline...
============================================================
📝 Step 1: Extracting keywords with TF-IDF + MMR...
🔍 Step 2: Searching multiple sources...
🧹 Step 3: AI-domain filtering...
📊 Step 4: Computing CBF similarity and ranking...
📌 Extracted keywords: ['transformer', 'attention', 'backpropagation', ...]
📥 Found 399 resources

📊 Full Pipeline Results:
  total_resources: 399
  ai_relevance: 91.0
  cross_platform_diversity: 5
  authority_score: 65.0
  url_validation: {'valid': 376, 'invalid': 23, 'valid_percentage': 94.2}

============================================================
COMPARISON: Restricted vs Full Pipeline
============================================================

🚀 Key Improvements:
  ai_relevance_delta: +59.0
  noise_reduction: 68.0
  diversity_delta: +2.0

💾 Results saved to: test/evaluation_pipeline/results/evaluation_results.json
```

### JSON 输出

```json
{
  "baseline": {
    "total_resources": 1247,
    "ai_relevance": 32.0,
    "cross_platform_diversity": 3,
    "authority_score": 28.0
  },
  "full_pipeline": {
    "total_resources": 399,
    "ai_relevance": 91.0,
    "cross_platform_diversity": 5,
    "authority_score": 65.0,
    "noise_reduction": 68.0
  },
  "improvements": {
    "ai_relevance_delta": 59.0,
    "noise_reduction": 68.0,
    "diversity_delta": 2.0
  }
}
```

## 🧪 测试用例

### 准备测试数据

创建不同主题的测试语料：

```bash
data/test_corpus/
├── neural_networks/
│   ├── nn_notes_1.txt
│   ├── nn_notes_2.txt
│   └── ...
├── nlp/
│   ├── nlp_lecture_1.txt
│   └── ...
└── computer_vision/
    └── ...
```

### 运行评估

```bash
# 运行单个主题评估
python test/evaluation_pipeline/evaluator.py --corpus data/test_corpus/neural_networks

# 运行完整评估
python test/evaluation_pipeline/evaluator.py --corpus data/test_corpus
```

## 📊 可视化结果

评估完成后，可以使用 Python 脚本生成图表：

```python
import json
import matplotlib.pyplot as plt

# 加载结果
with open('test/evaluation_pipeline/results/evaluation_results.json', 'r') as f:
    results = json.load(f)

# 绘制对比图
metrics = ['ai_relevance', 'cross_platform_diversity', 'authority_score']
baseline = [results['baseline'][m] for m in metrics]
full = [results['full_pipeline'][m] for m in metrics]

# 生成图表...
```

## 🔄 自动化测试

### 集成到 CI/CD

```yaml
# .github/workflows/evaluation.yml
name: Evaluation Tests
on: [push, pull_request]
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run evaluation
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python test/evaluation_pipeline/evaluator.py
```

## 📝 论文数据

本评估系统生成的数据可以直接用于论文：

1. **Table 1**: Keyword Extraction Quality
2. **Table 2**: AI-Domain Filtering Effectiveness
3. **Table 3**: Value of Full Pipeline
4. **Table 4**: Ablation Study: Component Contributions
5. **Figure 1**: Pipeline Comparison Chart

## 🐛 故障排查

### 问题 1: 导入错误

**症状**: `ModuleNotFoundError: No module named 'backend.core...'`

**解决方案**:
```bash
cd /path/to/AI-Pedia/Project
export PYTHONPATH=/path/to/AI-Pedia/Project:$PYTHONPATH
python test/evaluation_pipeline/evaluator.py
```

### 问题 2: 测试语料未找到

**症状**: `⚠️  No documents found in corpus!`

**解决方案**:
```bash
# 检查语料路径
ls -la data/test_corpus/*.txt

# 确保文件存在且为 .txt 格式
```

### 问题 3: 搜索失败

**症状**: `Warning: Search failed for keyword 'xxx'`

**解决方案**:
- 检查网络连接
- 确认搜索模块配置正确
- 查看详细错误日志

## 📞 支持

如有问题，请：
1. 查看本文档的故障排查部分
2. 检查 `test/evaluation_pipeline/results/` 中的详细日志
3. 提交 Issue 到项目仓库

---

## 📚 参考文献

本评估方法基于以下设计原则：

1. **Within-system evaluation**: 消除不同模型间的混淆变量
2. **Quantitative metrics**: 使用可量化的客观指标
3. **Ablation study**: 隔离各组件的贡献
4. **Reliability focus**: 强调搜索优先的可靠性优势

详见论文 Evaluation 章节。