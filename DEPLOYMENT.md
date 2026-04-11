# AI-Pedia 部署指南

## 📋 前提条件

- Docker 20.10+
- Docker Compose 1.29+
- 至少 2GB 可用内存
- （可选）OpenAI API Key 用于摘要生成功能

## 🚀 快速部署

### 方式一：使用 Docker Compose（推荐）

1. **克隆项目**
```bash
git clone <repository-url>
cd AI-Pedia/Project
```

2. **配置环境变量**
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，设置你的 OpenAI API Key（可选）
nano .env
```

3. **启动服务**
```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f ai-pedia
```

4. **访问应用**
打开浏览器访问：`http://localhost:5000`

5. **停止服务**
```bash
docker-compose down
```

### 方式二：使用 Docker 直接构建

```bash
# 1. 构建镜像
docker build -t ai-pedia:latest .

# 2. 运行容器
docker run -d \
  --name ai-pedia \
  -p 5000:5000 \
  -e OPENAI_API_KEY=your_api_key_here \
  -v $(pwd)/data:/app/data \
  ai-pedia:latest

# 3. 查看日志
docker logs -f ai-pedia

# 4. 停止容器
docker stop ai-pedia
docker rm ai-pedia
```

### 方式三：本地部署（开发环境）

1. **安装 Python 依赖**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

2. **设置环境变量**
```bash
export OPENAI_API_KEY=your_api_key_here
export FLASK_ENV=production
```

3. **启动应用**
```bash
python app.py
```

访问：`http://localhost:5000`

## 🔧 配置说明

### 环境变量

| 变量名 | 必需 | 默认值 | 说明 |
|--------|------|--------|------|
| `OPENAI_API_KEY` | 否 | - | OpenAI API Key，用于摘要生成。不提供则使用规则回退 |
| `FLASK_ENV` | 否 | `production` | Flask 运行环境 |
| `FLASK_DEBUG` | 否 | `False` | Flask 调试模式 |
| `FLASK_PORT` | 否 | `5000` | 服务端口 |

### Pipeline 参数

在 `app.py` 中可以调整以下参数：

```python
# 关键词提取数量
DEFAULT_KEYWORD_COUNT = 10

# CBF 相似度阈值
SIMILARITY_THRESHOLD = 0.05

# 每种类型返回的资源数量
TOP_K_RESOURCES = 5
```

## 📊 监控和维护

### 健康检查

应用提供 `/health` 端点用于健康检查：

```bash
curl http://localhost:5000/health
```

响应：
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "keyword_extractor": "ok",
    "resource_searcher": "ok",
    "recommender": "ok"
  }
}
```

### 查看日志

**Docker Compose:**
```bash
docker-compose logs -f ai-pedia
```

**Docker:**
```bash
docker logs -f ai-pedia
```

### 备份数据

数据存储在 `data/` 目录：

```bash
# 备份
tar -czf ai-pedia-backup-$(date +%Y%m%d).tar.gz data/

# 恢复
tar -xzf ai-pedia-backup-YYYYMMDD.tar.gz
```

## 🔒 安全建议

1. **保护 API Key**
   - 不要在代码中硬编码 API Key
   - 使用环境变量或密钥管理服务
   - 定期轮换 API Key

2. **网络安全**
   - 在生产环境中使用反向代理（Nginx）
   - 启用 HTTPS
   - 限制上传文件大小

3. **文件权限**
   - 确保 `.env` 文件权限为 `600`
   - 定期清理 `data/uploads` 目录

## 🐛 故障排查

### 问题 1: 容器启动失败

**症状：** `docker-compose up` 后容器立即退出

**解决方案：**
```bash
# 查看详细日志
docker-compose logs ai-pedia

# 检查端口是否被占用
netstat -an | grep 5000

# 尝试使用不同端口
docker run -d -p 8080:5000 ai-pedia:latest
```

### 问题 2: OpenAI API 调用失败

**症状：** 摘要生成失败，回退到规则生成

**解决方案：**
```bash
# 检查环境变量是否正确设置
docker-compose exec ai-pedia env | grep OPENAI

# 验证 API Key 格式
# 应该是：sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 问题 3: 上传文件后处理失败

**症状：** 上传进度显示但无结果

**解决方案：**
```bash
# 检查容器内文件权限
docker-compose exec ai-pedia ls -la data/uploads/

# 重启服务
docker-compose restart
```

## 📈 性能优化

### 增加并发处理

修改 `app.py` 中的线程池大小：

```python
# 在 app.py 开头添加
from concurrent.futures import ThreadPoolExecutor

# 创建线程池
executor = ThreadPoolExecutor(max_workers=4)

# 在处理函数中使用
executor.submit(process_documents, zip_path)
```

### 使用 Redis 缓存

对于频繁查询的资源，可以添加 Redis 缓存：

```bash
# docker-compose.yml 中添加
services:
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

## 🔄 更新部署

```bash
# 1. 拉取最新代码
git pull origin main

# 2. 重新构建镜像
docker-compose build

# 3. 重启服务（保留数据）
docker-compose up -d

# 4. 清理旧镜像
docker image prune -a
```

## 📞 支持

如有问题，请：
1. 查看日志文件
2. 检查健康检查端点
3. 参考本文档的故障排查部分

---

## 📝 附录

### 目录结构

```
AI-Pedia/Project/
├── app.py                 # Flask 主应用
├── requirements.txt        # Python 依赖
├── Dockerfile             # Docker 镜像定义
├── docker-compose.yml     # Docker Compose 配置
├── .env.example           # 环境变量模板
├── .dockerignore          # Docker 忽略文件
├── backend/               # 后端核心模块
│   ├── core/             # 核心功能
│   │   ├── keyword_extractor.py
│   │   ├── resource_searcher.py
│   │   ├── recommender.py
│   │   └── ai_summarizer.py
│   └── utils/            # 工具函数
├── frontend/             # 前端模板和静态资源
│   ├── templates/        # HTML 模板
│   └── static/           # CSS/JS/图片
├── data/                 # 数据目录
│   ├── uploads/          # 上传的文件
│   ├── results/          # 处理结果
│   └── outputs/          # 最终输出
└── test/                 # 测试和评估
    └── evaluation_pipeline/
```

### 端口说明

- `5000`: AI-Pedia Web 服务

### 网络配置

默认使用桥接网络 `ai-pedia-network`。如需自定义，修改 `docker-compose.yml`：

```yaml
networks:
  ai-pedia-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```