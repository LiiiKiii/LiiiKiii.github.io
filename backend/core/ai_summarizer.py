#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI摘要生成模块
使用AI API为资源生成智能摘要
"""

import os
import re
from typing import Dict, Optional

# 尝试导入OpenAI
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# 尝试导入其他AI库作为fallback
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


OPENAI_SUMMARY_MODEL = os.environ.get("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")


def get_openai_api_key(api_key_from_request: Optional[str] = None) -> Optional[str]:
    """
    获取OpenAI API Key
    优先级：请求中的key > 环境变量
    
    Args:
        api_key_from_request: 从请求中传递的API key（可选）
    
    Returns:
        API key字符串，如果不存在则返回None
    """
    # 优先使用请求中传递的key
    if api_key_from_request:
        return api_key_from_request
    
    # 否则从环境变量获取
    return os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")


def generate_summary_with_openai(content: str, resource_type: str, title: str = "", max_tokens: int = 150, api_key: Optional[str] = None) -> Optional[str]:
    """
    使用OpenAI API生成摘要
    
    Args:
        content: 资源内容
        resource_type: 资源类型 ('txt', 'video', 'code')
        title: 资源标题
        max_tokens: 最大token数
        api_key: OpenAI API key（可选，如果不提供则从环境变量获取）
    
    Returns:
        摘要字符串，如果失败返回None
    """
    if not HAS_OPENAI:
        return None
    
    # 获取API key（优先使用传入的key）
    api_key = api_key or get_openai_api_key()
    if not api_key:
        return None
    
    try:
            # 根据资源类型和内容长度构建提示词
        if resource_type == "txt":
            prompt = f"""请为以下学术资源生成一个简洁的简介，包含三个方面：

1. 这是关于什么的：用一句话说明这个资源的核心主题和内容
2. 有什么亮点：简要说明这个资源的重点或特色
3. 你能学到什么：说明通过这个资源可以掌握什么知识或技能

要求：
- 总共3-4句话，简洁明了
- 不要重复标题
- 不要使用"与您的学习资料相关"等模糊表述
- 直接说明内容，让用户快速了解资源的价值

示例格式：
"这是一篇关于训练集、验证集和测试集的学术文章。文章详细介绍了这三种数据集在机器学习中的作用和区别，以及如何正确划分数据集来评估模型性能。通过阅读这篇文章，你将掌握数据集划分的最佳实践，理解如何避免过拟合和欠拟合问题，并学会使用交叉验证等评估方法。"

标题: {title}

内容片段:
{content}

简介:"""
        elif resource_type == "video":
            prompt = f"""请为以下视频资源生成一个2-3句话的简介，帮助用户快速了解这个视频的内容。

要求：
1. 第一句：说明视频类型和来源（如"这是一个来自YouTube的教学视频"）
2. 第二句：直接说明视频讲了什么内容、教授什么知识点、涵盖什么主题
3. 第三句（可选）：如果内容较复杂，可以补充一个关键点或适用场景
4. 总共2-3句话，简洁明了，不要冗长
5. 不要重复标题
6. 不要使用"与您的学习资料相关"、"这个视频"等模糊表述
7. 直接说明内容，让用户快速知道这个视频是关于什么的

示例格式：
"这是一个来自YouTube的教学视频。视频讲解了循环神经网络(RNN)的基本原理，包括LSTM和GRU的结构，并通过实例演示了如何使用PyTorch实现RNN模型。适合初学者学习深度学习中的序列模型。"

标题: {title}

描述: {content}

简介:"""
        elif resource_type == "code":
            prompt = f"""请为以下代码资源生成一个2-3句话的简介，帮助用户快速了解这个代码库的内容。

要求：
1. 第一句：说明代码库类型和来源（如"这是一个来自GitHub的开源项目"）
2. 第二句：直接说明这个代码库实现了什么功能、使用了什么技术、解决了什么问题
3. 第三句（可选）：如果内容较复杂，可以补充一个关键特性或应用场景
4. 总共2-3句话，简洁明了，不要冗长
5. 不要重复标题
6. 不要使用"与您的学习资料相关"、"这个代码"等模糊表述
7. 直接说明内容，让用户快速知道这个代码库是关于什么的

示例格式：
"这是一个来自GitHub的开源项目。项目实现了基于PyTorch的GAN生成对抗网络，包含DCGAN和WGAN两种模型，提供了完整的训练脚本和预训练权重。可用于图像生成任务和GAN模型研究。"

标题: {title}

描述: {content}

简介:"""
        
        # 尝试使用新版本OpenAI库（v1.0+）
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=OPENAI_SUMMARY_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个专业的学术资源推荐助手。请生成简洁的简介，包含三个方面：1)这是关于什么的，2)有什么亮点，3)你能学到什么。要求简洁明了，不要冗长，不要使用'与您的学习资料相关'等模糊表述。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                timeout=10
            )
            summary = response.choices[0].message.content.strip()
        except (ImportError, AttributeError):
            # 回退到旧版本API
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model=OPENAI_SUMMARY_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个专业的学术资源推荐助手。请生成简洁的简介，包含三个方面：1)这是关于什么的，2)有什么亮点，3)你能学到什么。要求简洁明了，不要冗长，不要使用'与您的学习资料相关'等模糊表述。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                timeout=10
            )
            summary = response.choices[0].message.content.strip()
        
        # 清理摘要：移除可能的引号、多余空格等
        summary = re.sub(r'^["\']|["\']$', '', summary)
        summary = re.sub(r'\s+', ' ', summary).strip()
        
        return summary if summary else None
        
    except Exception as e:
        return None


def generate_summary_with_fallback(content: str, resource_type: str, title: str = "") -> str:
    """
    使用fallback方法生成摘要（当没有AI API或API失败时）
    基于规则和启发式方法生成摘要，生成2-3句话的简介
    
    Args:
        content: 资源内容
        resource_type: 资源类型
        title: 资源标题
    
    Returns:
        摘要字符串
    """
    # 清理内容
    content = re.sub(r'<[^>]+>', '', content)  # 移除HTML标签
    content = re.sub(r'\s+', ' ', content).strip()
    
    if resource_type == "txt":
        # 提取前2-3个有意义的句子，确保是对内容的介绍
        sentences = re.split(r'[.!?]\s+', content)
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
        
        if meaningful_sentences:
            # 取前2-3句，但避免重复标题和无关的元信息
            summary_parts = []
            for sent in meaningful_sentences[:3]:
                # 跳过与标题高度相似的句子
                if title and title.lower() in sent.lower()[:50]:
                    continue
                # 跳过包含"From Wikipedia"、"Redirected from"等元信息的句子
                if any(phrase in sent for phrase in ["From Wikipedia", "Redirected from", "Part of a series"]):
                    continue
                # 跳过太短的句子
                if len(sent) < 40:
                    continue
                summary_parts.append(sent)
                # 限制在2-3句话，总长度不超过120字符
                if len(summary_parts) >= 3 or len('. '.join(summary_parts)) > 120:
                    break
            
            if summary_parts:
                summary = '. '.join(summary_parts)
                if not summary.endswith('.'):
                    summary += '.'
                # 确保摘要不超过120字符（2-3句话）
                if len(summary) > 120:
                    summary = summary[:117] + "..."
                return summary
        
        # 如果没有好的句子，尝试提取内容摘要（跳过开头可能的元信息）
        if content:
            # 移除标题和常见的元信息
            content_without_title = content
            if title:
                content_without_title = content.replace(title, '')
            # 移除开头的元信息行
            lines = content_without_title.split('\n')
            clean_lines = []
            skip_patterns = ["From Wikipedia", "Redirected from", "Part of a series", "Tasks in machine learning"]
            for line in lines:
                if not any(pattern in line for pattern in skip_patterns):
                    clean_lines.append(line)
                if len(' '.join(clean_lines)) > 100:
                    break
            clean_content = ' '.join(clean_lines).strip()
            if clean_content:
                return clean_content[:120] + "..." if len(clean_content) > 120 else clean_content
            # 如果清理后没有内容，返回原始内容的前120字符
            return content_without_title[:120] + "..." if len(content_without_title) > 120 else content_without_title
    
    elif resource_type == "video":
        if content:
            # 视频描述通常已经比较简洁
            desc = content[:120]
            if len(content) > 120:
                # 尝试在句号处截断
                sentences = desc.split('.')
                if len(sentences) > 1:
                    desc = '. '.join(sentences[:-1]) + '.'
            return desc
        return title if title else "视频资源"
    
    elif resource_type == "code":
        if content:
            # 代码资源描述
            desc = content[:100]
            return desc + "..." if len(content) > 100 else desc
        return title if title else "代码资源"
    


def extract_abstract_from_content(content: str) -> str:
    """
    从内容中提取abstract（摘要）
    支持多种格式：
    - "摘要: {abstract}\n\n"
    - "Abstract: {abstract}\n\n"
    - arXiv格式等
    
    Args:
        content: 资源内容
    
    Returns:
        提取的abstract，如果没有则返回空字符串
    """
    if not content:
        return ""
    
    # 尝试匹配"摘要:"或"Abstract:"开头的摘要
    # 支持多种格式：
    # 1. "摘要: {abstract}\n\n" (Google Scholar格式)
    # 2. "Abstract: {abstract}\n\n" (英文格式)
    # 3. "摘要: {abstract}论文链接" (带后续文本)
    patterns = [
        r'摘要[：:]\s*(.+?)(?:\n\n|论文链接|arXiv链接|请访问|$)',  # 中文格式，使用非贪婪匹配直到分隔符
        r'Abstract[：:]\s*(.+?)(?:\n\n|论文链接|arXiv链接|请访问|$)',  # 英文格式
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            abstract = match.group(1).strip()
            # 清理abstract：移除多余的换行和空格，但保留句子结构
            abstract = re.sub(r'\n+', ' ', abstract)  # 将多个换行替换为空格
            abstract = re.sub(r'\s+', ' ', abstract)  # 将多个空格替换为单个空格
            # 移除可能的作者信息（如果abstract中包含）
            abstract = re.sub(r'作者[：:][^。]+。', '', abstract)
            abstract = re.sub(r'Authors?[：:][^.]+\.[\s.]*', '', abstract, flags=re.IGNORECASE)
            abstract = abstract.strip()
            # 确保abstract有实际内容且不是太短
            if abstract and len(abstract) > 30:  # 至少30个字符
                return abstract
    
    return ""


def generate_simple_wikipedia_summary(content: str, title: str) -> str:
    """
    为Wikipedia资源生成简单的简介（只说明这是关于什么的）
    
    Args:
        content: 资源内容
        title: 资源标题
    
    Returns:
        简单简介字符串
    """
    # 只返回简单的说明，不包含后续内容
    return f"这是关于{title}的百科文章。"


def generate_resource_summary(resource: Dict, resource_type: str, openai_api_key: Optional[str] = None) -> Dict:
    """
    为资源生成智能摘要
    
    Args:
        resource: 资源字典
        resource_type: 资源类型 ('txt', 'video', 'code')
    
    Returns:
        包含summary和summary_type的字典
        summary_type: 'ai_generated' | 'abstract' | 'wikipedia_simple' | None
    """
    title = resource.get("title", "")
    content = resource.get("content", "")
    description = resource.get("description", "")
    url = resource.get("url", "")
    source = resource.get("source", "")
    
    # 对于文本资源，检查是否是ArXiv文章（有Abstract）
    if resource_type == "txt":
        abstract = extract_abstract_from_content(content)
        if abstract:
            # 检查是否是ArXiv相关来源
            is_arxiv = any(keyword in (source or "").lower() for keyword in ["arxiv", "google scholar", "scholar"])
            is_arxiv = is_arxiv or "arxiv" in (url or "").lower()
            
            if is_arxiv:
                # ArXiv文章：返回Abstract，标记为需要展开/收起
                return {
                    "summary": abstract,
                    "summary_type": "abstract"
                }
    
    # 准备内容文本
    content_text = ""
    if resource_type == "txt":
        content_text = content or description or ""
    elif resource_type == "video":
        content_text = description or content or ""
    elif resource_type == "code":
        content_text = description or content or ""
    
    # 对于Wikipedia等长文本，提取更多内容用于生成摘要
    # 但限制总长度避免API调用过长
    max_content_length = 2000 if "Wikipedia" in title or len(content_text) > 3000 else 1000
    if len(content_text) > max_content_length:
        content_text = content_text[:max_content_length] + "..."
    
    # 尝试使用OpenAI API生成摘要
    # 限制在100 tokens，生成更详细的简介（关于啥的、有啥亮点、你能学到啥）
    max_tokens = 100
    ai_summary = generate_summary_with_openai(content_text, resource_type, title, max_tokens=max_tokens, api_key=openai_api_key)
    if ai_summary:
        return {
            "summary": ai_summary,
            "summary_type": "ai_generated"
        }
    
    # OpenAI失败时，根据资源类型处理
    if resource_type == "txt":
        # 检查是否是Wikipedia
        is_wikipedia = "Wikipedia" in (title or "") or "wikipedia" in (url or "").lower() or "wikipedia" in (source or "").lower()
        
        if is_wikipedia:
            # Wikipedia：生成简单简介
            simple_summary = generate_simple_wikipedia_summary(content_text, title)
            return {
                "summary": simple_summary,
                "summary_type": "wikipedia_simple"
            }
        else:
            # 其他文本资源：返回None，不使用fallback
            return {
                "summary": None,
                "summary_type": None
            }
    else:
        # 视频和代码资源：OpenAI失败时返回None
        return {
            "summary": None,
            "summary_type": None
        }
