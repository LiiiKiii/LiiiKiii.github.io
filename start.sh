#!/bin/bash
# AI领域多媒体推荐系统 启动脚本

echo "=========================================="
echo "  AI领域多媒体推荐系统"
echo "=========================================="
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
python3 -c "import flask, numpy, sklearn, requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "警告: 部分依赖可能未安装"
    echo "正在安装依赖..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "错误: 依赖安装失败，请手动运行: pip3 install -r requirements.txt"
        exit 1
    fi
fi

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p data/uploads data/results data/outputs

# 设置OpenAI API Key（在检查之前设置）
export OPENAI_API_KEY="PUT YOUR OPENAI API KEY HERE"

# 检查OpenAI API Key配置
echo ""
if [ -z "$OPENAI_API_KEY" ]; then
    echo "提示: 未检测到 OPENAI_API_KEY 环境变量"
    echo "  - 如需使用LLM摘要功能，请设置API Key（可选）"
    echo "  - 设置方法1: 在终端运行: export OPENAI_API_KEY='your-key-here'"
    echo "  - 设置方法2: 编辑此脚本，在下方添加: export OPENAI_API_KEY='your-key-here'"
    echo "  - 如果没有API Key，系统会使用规则型fallback方法生成摘要"
    echo ""
else
    echo "✓ 检测到 OPENAI_API_KEY，将使用LLM生成摘要"
    echo ""
fi

# 提示信息
echo "系统使用通用搜索方式，无需其他API密钥配置"
echo "  文本搜索: Wikipedia / Google Scholar / arXiv 等学术资源"
echo "  视频搜索: YouTube HTML 解析"
echo "  图片搜索: Google Images / Bing Images / Unsplash / Pexels 等"
echo ""

# 启动应用
echo "启动应用..."
echo "访问地址: http://localhost:5000"
echo "按 Ctrl+C 停止服务"
echo ""

# 使用 env 命令显式传递环境变量，确保 Python 进程能正确继承
env OPENAI_API_KEY="$OPENAI_API_KEY" python3 app.py
