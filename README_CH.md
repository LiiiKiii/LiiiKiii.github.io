# AI领域多媒体推荐系统

一个专注于AI/机器学习领域的智能多媒体推荐系统，能够从用户上传的AI/ML文档中提取关键词，搜索外部资源（文本、视频、图片），并使用CBF（基于内容的推荐）算法筛选最相关的AI领域资源。

## 项目结构

```
Project/
├── app.py                      # Flask 主应用入口
├── requirements.txt            # Python 依赖
├── README.md / README_CH.md    # 中英文项目说明
├── start.sh                    # 启动脚本
├── index.html                  # 根目录跳转页（重定向到 frontend/，用于 GitHub Pages）
│
├── frontend/                   # 前端代码（本地 + GitHub Pages 共用）
│   ├── index.html              # 纯静态首页（GitHub Pages 使用）
│   ├── templates/              # Flask 模板（本地运行使用）
│   │   ├── index.html          # 首页模板
│   │   ├── help.html           # 帮助页面模板
│   │   ├── progress.html       # 研发进度页面模板
│   │   ├── ai-enhance.html     # AI 增强页面模板
│   │   └── contact.html        # 加入我们页面模板
│   └── static/                 # 静态资源
│       ├── css/
│       │   ├── style.css       # 全局样式
│       │   ├── help.css        # 帮助页样式
│       │   ├── progress.css    # 进度页样式
│       │   ├── ai-enhance.css  # AI 增强页样式
│       │   └── contact.css     # 加入我们页样式
│       ├── js/
│       │   ├── main.js         # 全局逻辑（上传、SSE、i18n、主题等）
│       │   ├── help.js         # 帮助页交互
│       │   ├── progress.js     # 进度页动画
│       │   ├── ai-enhance.js   # AI 增强页逻辑
│       │   └── contact.js      # 联系表单逻辑
│       └── images/
│           └── home-icon.png   # 网站图标
│
├── backend/                    # 后端代码
│   ├── core/                   # 核心功能模块
│   │   ├── __init__.py
│   │   ├── keyword_extractor.py    # 关键词提取
│   │   ├── resource_searcher.py    # 资源搜索
│   │   ├── recommender.py          # CBF 推荐系统
│   │   └── ai_summarizer.py        # AI 摘要生成
│   └── utils/                  # 工具模块
│       ├── __init__.py
│       └── file_utils.py       # 文件处理工具
│
├── data/                       # 数据目录（运行时自动创建子目录）
│   ├── uploads/                # 用户上传的文件
│   ├── results/                # 搜索结果
│   ├── outputs/                # 最终输出
│   ├── test_corpus/            # 旧版单语料测试输入
│   └── test_corpora/           # 当前多语料评估输入（3 组 focused corpora）
│
├── test/                       # 测试与评估
│   ├── README.md               # 测试总览与说明
│   ├── evaluation_pipeline/    # 当前可复现评估脚本与结果
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   ├── config.py
│   │   └── results/
│   └── performance/            # 本地确定性性能测试
│       └── run_performance_tests.py
│
└── docs/                       # 详细文档
    ├── STRUCTURE.md            # 代码结构说明
    ├── TECHNICAL_OVERVIEW.md   # 技术概览
    ├── EVALUATION.md           # 评估方案
    ├── INNOVATION.md           # 创新点说明
    └── REFERENCES.md           # 参考文献与资料
```

## 功能特点

1. **AI领域专用**：专注于AI/机器学习领域的资源推荐，智能过滤非AI相关内容
2. **文件夹上传**：支持上传包含至少10个txt或pdf文档的zip文件夹（PDF会自动转换为TXT）
3. **智能关键词提取**：自动从AI/ML文档中提取关键主题和关键词（使用TF-IDF和MMR算法）
4. **多源搜索**：
   - 文本资源：Wikipedia、Google Scholar、arXiv、学术网站等
   - 视频资源：YouTube（英文内容优先）
   - 代码资源：GitHub、Google Colab、Kaggle、Stack Overflow、Papers with Code等
5. **AI关键词过滤**：使用AI领域关键词列表（70+个AI/ML核心术语）过滤不相关内容
6. **CBF推荐系统**：基于内容相似度筛选最相关的AI资源（设置相似度阈值≥0.05）
7. **内容清理**：自动移除联系方式、部门信息、网址等无关信息
8. **结果打包**：自动将推荐结果打包成zip文件供下载

## 安装步骤

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行应用

```bash
python app.py
```

或使用启动脚本：

```bash
chmod +x start.sh
./start.sh
```

应用将在 `http://localhost:5000` 启动。

---

## 在线访问（GitHub Pages 预览）

本项目提供一个 **纯前端静态版本**，可以直接通过 GitHub Pages 访问界面效果（不需要本地运行 Flask）：

- 部署方式：将仓库推送到 GitHub，使用 `main` 分支的根目录作为 Pages 源
- 访问路径：
  - 根地址 `https://<你的用户名>.github.io/` 会通过根目录的 `index.html` **自动重定向** 到 `frontend/`
  - 实际静态站点入口为 `https://<你的用户名>.github.io/frontend/`

注意：

- 静态版本会正常展示页面、主题切换、多语言切换、动画效果等前端交互  
- 但依赖后端的功能（如 ZIP 上传、关键词提取、推荐计算、联系我们表单提交等）在 GitHub Pages 上不会真正执行，需要连接到你部署好的 Flask 后端

## 配置说明

### AI摘要功能（可选）

系统支持使用OpenAI API为每个推荐资源生成智能摘要。如果配置了OpenAI API Key，系统会使用AI生成更准确、更有信息量的摘要；如果没有配置，系统会使用智能fallback方法生成摘要。

#### 配置OpenAI API Key（可选）

1. 获取OpenAI API Key：访问 https://platform.openai.com/api-keys 注册并获取API Key

2. 设置环境变量：
   ```bash
   # Linux/Mac
   export OPENAI_API_KEY="your-api-key-here"
   
   # Windows
   set OPENAI_API_KEY=your-api-key-here
   ```

   或者在启动脚本中设置：
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   python app.py
   ```

3. 如果没有配置API Key，系统会自动使用fallback方法，仍然可以正常工作。

## 使用说明

### 1. 准备数据

准备一个包含至少10个txt文档的文件夹，然后将其压缩成zip格式。

### 2. 上传和处理

1. 打开浏览器访问 `http://localhost:5000`
2. 点击上传区域，选择你的zip文件
3. 系统会自动验证文件数量（至少10个txt）
4. 上传成功后，系统会自动开始处理

### 3. 下载结果

处理完成后，点击"下载推荐结果"按钮，系统会下载一个zip文件，包含：
- `txt/` - 推荐的文本资源（已过滤联系方式、部门信息等无关内容）
- `video/` - 推荐的视频链接
- `code/` - 推荐的代码资源（GitHub、Colab、Kaggle等）

## 技术说明

### 关键词提取算法

- 使用TF-IDF向量化文档
- 使用MMR（Maximal Marginal Relevance）确保关键词的多样性和代表性
- 默认提取10个AI相关关键词
- **智能过滤噪声短语**：
  - 过滤网址、邮箱、联系方式
  - 过滤大学域名（如 durham.ac.uk, .edu 等）
  - 过滤机构信息（Department of X, University of Y 等）
  - 过滤地址、电话号码等无关信息
  - 保留学术相关术语（如 "center of mass" 等）

### AI摘要生成

- **优先使用OpenAI API**：如果配置了API Key，使用配置的 OpenAI 聊天模型生成智能摘要
- **智能Fallback**：如果没有API Key或API调用失败，使用基于规则的智能摘要生成
- **摘要特点**：
  - 不重复标题，提供实质性内容概述
  - 针对不同资源类型（文本、视频、代码）生成针对性摘要
  - 长度控制在50-100字，简洁明了

### CBF推荐算法

- 将用户文档和外部资源转换为TF-IDF向量
- 计算余弦相似度
- 设置相似度阈值（≥0.05），只推荐相关资源
- 使用AI关键词列表进行二次过滤，确保资源与AI领域相关
- 选择相似度最高的资源（默认每种类型5个）

### AI关键词过滤机制

- 维护70+个AI领域核心关键词（机器学习、深度学习、神经网络等）
- 在搜索阶段过滤明显不相关的URL和标题
- 在推荐阶段检查资源是否包含AI相关关键词
- 过滤掉报告、政策文档、建筑问题等非学术内容

### 资源搜索策略

- **文本**：优先搜索Wikipedia、Google Scholar、arXiv等学术资源
- **视频**：通过YouTube HTML页面解析获取视频信息（优先英文内容）
- **代码**：从GitHub、Google Colab、Kaggle、Stack Overflow、Papers with Code等平台搜索代码资源
- **AI关键词过滤**：在搜索和推荐阶段使用70+个AI领域关键词进行相关性过滤

**重要**：
- 本系统使用通用搜索方式，**不需要任何API密钥**！
- **AI摘要功能（可选）**：如需使用AI生成资源摘要，需要配置OpenAI API Key（见下方配置说明）。如果没有配置，系统会使用智能fallback方法生成摘要。

## 注意事项

1. **文档主题**：系统专为AI/机器学习领域设计，建议上传AI相关文档以获得最佳推荐效果
2. **文件数量要求**：上传的文件夹必须包含至少10个txt/pdf文档
3. **文件大小限制**：默认最大上传500MB
4. **处理时间**：根据文档数量和网络情况，处理可能需要几分钟
5. **网络要求**：需要稳定的网络连接以访问外部资源
6. **内容过滤**：系统会自动过滤非AI相关内容（如报告、政策文档、建筑问题等）
7. **相似度阈值**：只推荐相似度≥0.05的资源，确保推荐内容的相关性

## 故障排除

### 上传失败
- 确保文件是zip格式
- 确保zip内包含至少10个txt文件
- 检查文件大小是否超过限制

### 处理失败
- 检查网络连接（需要访问外部资源）
- 查看控制台错误信息
- 确保所有依赖已正确安装

## 开发说明

### 扩展搜索源

在 `backend/core/resource_searcher.py` 中添加新的搜索函数。

### 调整推荐算法

修改 `backend/core/recommender.py` 中的相似度计算方法。

### 自定义关键词数量

在 `app.py` 的 `extract_keywords_from_folder` 调用中修改 `top_k` 参数。

## 许可证

本项目仅供学习和研究使用。
