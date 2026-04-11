# 项目代码结构说明

## 目录结构

```
Project/
├── app.py                      # 主应用入口文件（路由与 SSE 编排）
├── config.py                   # 路径与可调参数（上传上限、关键词数量、搜索上限等）
├── requirements.txt            # Python依赖包列表
├── README.md                   # 项目说明文档
├── start.sh                    # 启动脚本
│
├── frontend/                   # 前端文件目录
│   ├── templates/             # HTML模板
│   │   ├── index.html         # 主页面（文件上传和处理）
│   │   ├── help.html          # 帮助页面
│   │   ├── progress.html      # 研发进度页面
│   │   ├── ai-enhance.html    # AI增强页面
│   │   └── contact.html       # 联系我们页面
│   └── static/                 # 静态资源
│       ├── css/
│       │   ├── style.css      # 全局样式文件
│       │   ├── help.css       # 帮助页面样式
│       │   ├── progress.css   # 进度页面样式
│       │   ├── ai-enhance.css # AI增强页面样式
│       │   └── contact.css    # 联系我们页面样式
│       └── js/
│           ├── main.js        # 全局JavaScript文件（包含i18n、主题切换等）
│           ├── help.js        # 帮助页面逻辑
│           ├── progress.js    # 进度页面逻辑
│           ├── ai-enhance.js  # AI增强页面逻辑
│           └── contact.js     # 联系我们页面逻辑
│
├── backend/                    # 后端代码目录
│   ├── core/                   # 核心功能模块
│   │   ├── __init__.py
│   │   ├── keyword_extractor.py    # 关键词提取模块
│   │   ├── resource_searcher.py   # 资源搜索模块
│   │   ├── recommender.py          # CBF推荐系统
│   │   └── ai_summarizer.py        # AI摘要生成模块
│   └── utils/                  # 工具模块
│       ├── __init__.py
│       ├── file_utils.py       # 文件处理工具
│       └── search_persist.py   # 将原始搜索结果落盘到 data/results/
│
├── data/                       # 数据目录
│   ├── uploads/               # 用户上传的文件（自动创建）
│   ├── results/               # 搜索结果（自动创建）
│   └── outputs/                # 最终输出（自动创建）
│
└── docs/                       # 文档目录
    └── STRUCTURE.md           # 本文件
```

## 模块说明

### 前端 (frontend/)

#### 页面模板 (templates/)
- **index.html**: 主页面，包含文件上传、处理进度、结果展示
- **help.html**: 帮助页面，介绍产品功能、技术原理、创新理念
- **progress.html**: 研发进度页面，展示项目开发时间线和功能模块状态
- **ai-enhance.html**: AI增强页面，提供OpenAI API密钥管理和功能说明
- **contact.html**: 联系我们页面，包含贡献者信息和表单提交

#### 样式文件 (static/css/)
- **style.css**: 全局样式，包含导航栏、主题切换、响应式布局、移动端适配
- **help.css**: 帮助页面专用样式（轮播、卡片、模态框）
- **progress.css**: 进度页面专用样式（时间线、模块卡片）
- **ai-enhance.css**: AI增强页面专用样式
- **contact.css**: 联系我们页面专用样式

#### JavaScript文件 (static/js/)
- **main.js**: 全局JavaScript，包含：
  - 国际化（i18n）实现：`I18N_MAP`、`applyLanguage()`、`getCurrentLanguage()`
  - 主题切换：亮色/暗色模式
  - 文件上传和处理逻辑
  - SSE实时进度更新
  - 资源展示和下载
  - 导航栏功能
- **help.js**: 帮助页面逻辑（轮播、模态框、滚动动画）
- **progress.js**: 进度页面逻辑（滚动渐入动画）
- **ai-enhance.js**: AI增强页面逻辑（API密钥管理、滚动动画）
- **contact.js**: 联系我们页面逻辑（表单提交、滚动动画）

### 后端核心模块 (backend/core/)

#### keyword_extractor.py
- 功能：从文档中提取关键词/主题
- 主要函数：
  - `extract_keywords_from_folder()`: 从文件夹提取关键词
  - `basic_clean()`: 文本清洗
  - `mmr_select()`: MMR算法选择关键词

#### resource_searcher.py
- 功能：搜索外部资源（文本、视频、代码）
- 主要函数：
  - `search_text_resources()`: 搜索文本资源（Wikipedia、Google Scholar、arXiv）
  - `search_youtube_videos()`: 搜索YouTube视频
  - `search_code_resources()`: 搜索代码资源（GitHub）
  - `search_all_resources()`: 搜索所有类型资源
  - `filter_english_content()`: 过滤非英文内容
  - `is_irrelevant_url()`: 判断URL是否不相关
  - `clean_extracted_content()`: 清理提取的内容

#### recommender.py
- 功能：CBF推荐系统，基于相似度筛选资源
- 主要函数：
  - `recommend_best_resources()`: 推荐最佳资源
  - `compute_similarity()`: 计算相似度
  - `save_recommended_resources()`: 保存推荐结果

#### ai_summarizer.py
- 功能：为推荐资源生成智能摘要
- 主要函数：
  - `generate_resource_summary()`: 生成资源摘要（优先使用OpenAI，失败则使用fallback）
  - `generate_summary_with_openai()`: 使用OpenAI API生成摘要
  - `generate_summary_with_fallback()`: 基于规则的fallback摘要生成
  - `generate_wikipedia_summary()`: 为Wikipedia资源生成简单摘要

### 工具模块 (backend/utils/)

#### file_utils.py
- 功能：文件处理工具
- 主要函数：
  - `count_txt_files()`: 统计txt文件数量
  - `extract_zip()`: 解压zip文件
  - `create_output_zip()`: 创建zip文件
  - `sanitize_filename()`: 清理文件名

### 主应用 (app.py)

- Flask应用主文件
- 路由：
  - `GET /`: 主页
  - `GET /help`: 帮助页面
  - `GET /progress`: 研发进度页面
  - `GET /ai-enhance`: AI增强页面
  - `GET /contact`: 联系我们页面
  - `POST /upload`: 上传ZIP文件
  - `POST /process`: 处理文件夹（SSE流式返回进度）
  - `GET /download/<folder_name>`: 下载推荐结果ZIP文件
  - `GET /status/<folder_name>`: 获取处理状态
  - `POST /contact`: 提交联系我们表单
  - `POST /cleanup/<folder_name>`: 清理用户数据

## 数据流

1. **上传阶段**: 用户上传zip文件 → `data/uploads/`
2. **处理阶段**: 
   - 提取关键词 → 搜索外部资源 → `data/results/`
   - CBF推荐 → `data/outputs/`
3. **下载阶段**: 用户下载相应推荐结果zip文件

## 使用方式

1. 安装依赖：`pip install -r requirements.txt`
2. 启动应用：`python app.py` 或 `./start.sh`
3. 访问：`http://localhost:5000`

## 核心功能特性

### 国际化（i18n）
- 支持中英文界面切换
- 使用 `data-i18n-key` 属性标记需要翻译的元素
- `main.js` 中的 `I18N_MAP` 包含所有翻译文本
- 语言偏好保存在 `localStorage` 中

### 主题切换
- 支持亮色/暗色模式切换
- 使用CSS变量和类切换实现
- 主题偏好保存在 `localStorage` 中

### 响应式设计
- 移动端适配（`@media (max-width: 768px)` 和 `@media (max-width: 480px)`）
- 移动端导航栏切换为下拉菜单
- 所有页面都支持响应式布局

### 实时进度更新
- 使用Server-Sent Events (SSE) 实现实时进度反馈
- 显示处理步骤、关键词搜索进度、资源统计等

## 注意事项

- `data/` 目录下的文件会在运行时自动创建
- 所有模块都使用相对导入，通过 `sys.path` 配置
- 前端静态文件路径在Flask配置中指定
- 国际化翻译文本集中在 `main.js` 的 `I18N_MAP` 对象中
- 移动端适配主要通过CSS媒体查询实现

