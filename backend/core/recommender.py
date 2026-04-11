#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于内容的推荐系统（CBF - Content-Based Filtering）
通过相似度计算筛选最佳资源
"""

import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

# 导入清理函数
try:
    from backend.core.resource_searcher import clean_extracted_content, clean_title
except ImportError:
    # 如果导入失败（可能是循环导入），定义简单的清理函数
    def clean_extracted_content(content: str) -> str:
        """简单的清理函数，如果无法导入主要函数则使用此函数"""
        return content
    
    def clean_title(title: str) -> str:
        """简单的标题清理函数"""
        return title

def read_txt_files(folder_path: str) -> List[str]:
    """
    读取文件夹中所有txt文件的内容
    返回: List[str] 每个元素是一个文档的文本内容
    排除macOS系统文件（以._开头的资源分叉文件）和其他隐藏文件
    """
    texts = []
    if not os.path.isdir(folder_path):
        return texts
    
    for root, dirs, files in os.walk(folder_path):
        for fname in files:
            # 过滤掉macOS资源分叉文件（以._开头）和其他系统隐藏文件
            if fname.startswith('._') or fname.startswith('.DS_Store'):
                continue
            if fname.lower().endswith(".txt"):
                file_path = os.path.join(root, fname)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().strip()
                        if content:
                            texts.append(content)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return texts


def clean_text(text: str) -> str:
    """基础文本清洗"""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\u2000-\u206F\u2E00-\u2E7F\'\"\"''',.:;!?()[\]{}<>~`•…–—/_+=*^%$#@\\|-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_relevant_resource(resource: Dict, user_docs: List[str]) -> bool:
    """
    检查资源是否与用户文档相关
    过滤掉明显不相关的资源（如报告、政策文档等与学习材料无关的内容）
    
    Args:
        resource: 资源字典
        user_docs: 用户文档列表
    
    Returns:
        True 如果资源相关，False 否则
    """
    # 不相关的关键词模式（如果资源包含这些且不包含学术关键词，可能不相关）
    irrelevant_patterns = [
        r'\braac\b',  # RAAC报告（建筑问题）
        r'\breport\s+(on|about|of)\s+(the|impact|disruption)',  # 报告类文档
        r'\bpandemic\s+impact',  # 疫情影响报告
        r'\bschool\s+(closure|building|infrastructure)',  # 学校建筑问题
        r'\bcovid-?19\s+(impact|effect)',  # 新冠疫情影响
        r'\bministerial\s+(visit|report)',  # 部长访问报告
    ]
    
    # 导入AI相关关键词列表
    try:
        from backend.core.resource_searcher import AI_RELEVANT_KEYWORDS as academic_keywords
    except ImportError:
        # 如果导入失败，使用简化的关键词列表
        academic_keywords = [
            'machine learning', 'deep learning', 'neural network', 'algorithm',
            'model', 'training', 'data', 'classification', 'regression',
            'supervised', 'unsupervised', 'reinforcement', 'gradient',
            'optimization', 'loss function', 'activation', 'backpropagation',
            'convolutional', 'recurrent', 'transformer', 'attention',
            'natural language processing', 'computer vision', 'speech recognition',
            'artificial intelligence', 'ai', 'ml', 'dl', 'nlp', 'cv',
        ]
    
    # 检查标题和URL
    title = resource.get("title", "").lower()
    url = resource.get("url", "").lower()
    content = resource.get("content", "").lower()
    description = resource.get("description", "").lower()
    
    combined_text = f"{title} {url} {content[:500]} {description}"  # 只检查前500字符的内容
    
    # 检查是否包含明显不相关的模式
    for pattern in irrelevant_patterns:
        if re.search(pattern, combined_text, re.IGNORECASE):
            # 如果包含不相关模式，检查是否也包含学术关键词
            has_academic_keywords = any(keyword in combined_text for keyword in academic_keywords)
            if not has_academic_keywords:
                # 包含不相关模式且没有学术关键词，判定为不相关
                print(f"  过滤不相关资源: {resource.get('title', 'Unknown')[:50]} (包含不相关模式)")
                return False
    
    # 检查URL中是否包含明显不相关的路径（如报告、政策等）
    irrelevant_url_patterns = [
        r'/report', r'/policy', r'/impact', r'/disruption',
        r'raac', r'building', r'infrastructure', r'school-closure',
    ]
    
    for pattern in irrelevant_url_patterns:
        if re.search(pattern, url, re.IGNORECASE):
            # 如果URL包含不相关模式，检查是否也包含学术关键词
            has_academic_keywords = any(keyword in combined_text for keyword in academic_keywords)
            if not has_academic_keywords:
                print(f"  过滤不相关资源: {resource.get('title', 'Unknown')[:50]} (URL包含不相关路径)")
                return False
    
    # 放宽条件：如果资源包含任何AI相关关键词，就认为相关
    if any(keyword in combined_text for keyword in academic_keywords):
        return True
    
    # 如果资源标题或描述中包含学习相关内容，也认为相关
    learning_keywords = ['learning', 'tutorial', 'course', 'lecture', 'guide', 'introduction', 
                         'explained', 'overview', 'basics', 'fundamentals', 'concept', 'theory',
                         'neural', 'network', 'model', 'algorithm', 'data', 'training']
    if any(keyword in combined_text for keyword in learning_keywords):
        return True
    
    # 默认认为相关（放宽条件，确保有推荐结果）
    return True


def compute_domain_bonus(resource: Dict) -> float:
    """为明显更贴近AI学习场景的资源提供轻量级重排加分。"""
    try:
        from backend.core.resource_searcher import AI_RELEVANT_KEYWORDS as ai_keywords
    except ImportError:
        ai_keywords = [
            'machine learning', 'deep learning', 'neural network', 'transformer',
            'attention', 'computer vision', 'natural language processing', 'reinforcement learning',
            'pytorch', 'tensorflow', 'github', 'arxiv', 'wikipedia'
        ]

    title = resource.get("title", "").lower()
    description = resource.get("description", "").lower()
    url = resource.get("url", "").lower()
    source = resource.get("source", "").lower()
    combined = f"{title} {description} {url} {source}"

    bonus = 0.0
    if any(keyword in combined for keyword in ai_keywords):
        bonus += 0.05
    if any(authority in url for authority in ["arxiv.org", "wikipedia.org", "scholar.google.com", "github.com"]):
        bonus += 0.03
    if any(keyword in title for keyword in ai_keywords):
        bonus += 0.02
    return bonus


def generate_resource_summary_legacy(resource: Dict, resource_type: str) -> str:
    """
    为资源生成内容摘要，让用户了解资源的大概内容
    
    Args:
        resource: 资源字典
        resource_type: 资源类型 ('txt', 'video', 'code')
    
    Returns:
        内容摘要字符串
    """
    title = resource.get("title", "")
    content = resource.get("content", "")
    description = resource.get("description", "")
    url = resource.get("url", "")
    source = resource.get("source", "")
    
    # 合并所有可用文本
    all_text = ""
    if title:
        all_text += title + ". "
    if description:
        all_text += description + ". "
    if content:
        all_text += content
    
    # 清理文本：移除HTML标签、多余空格等
    import re
    all_text = re.sub(r'<[^>]+>', '', all_text)  # 移除HTML标签
    all_text = re.sub(r'\s+', ' ', all_text).strip()  # 规范化空格
    
    if resource_type == "txt":
        # 文本资源：提取内容的前2-3句话作为摘要
        if content:
            # 尝试按句子分割
            sentences = re.split(r'[.!?]\s+', content)
            # 过滤掉太短的句子
            meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
            if meaningful_sentences:
                # 取前2-3句，总长度不超过200字符
                summary_parts = []
                total_length = 0
                for sent in meaningful_sentences[:3]:
                    if total_length + len(sent) < 200:
                        summary_parts.append(sent)
                        total_length += len(sent) + 2
                    else:
                        break
                if summary_parts:
                    summary = ". ".join(summary_parts)
                    if summary and not summary.endswith('.'):
                        summary += "."
                    return summary[:200] + "..." if len(summary) > 200 else summary
        
        # 如果没有内容，使用标题和描述
        if title and description:
            return f"{title}. {description[:150]}..." if len(description) > 150 else f"{title}. {description}"
        elif title:
            return title
        else:
            return "文本资源"
    
    elif resource_type == "video":
        # 视频资源：使用描述或标题生成摘要
        if description:
            # 取描述的前150字符
            desc = description.strip()
            if len(desc) > 150:
                # 尝试在句号处截断
                sentences = desc.split('.')
                summary_parts = []
                total_length = 0
                for sent in sentences:
                    if total_length + len(sent) < 150:
                        summary_parts.append(sent.strip())
                        total_length += len(sent) + 1
                    else:
                        break
                if summary_parts:
                    summary = ". ".join(summary_parts)
                    if summary and not summary.endswith('.'):
                        summary += "."
                    return summary[:150] + "..." if len(summary) > 150 else summary
            return desc
        elif title:
            return title
        else:
            return "视频资源"
    
    elif resource_type == "code":
        # 代码资源：根据描述、标题和来源生成摘要
        summary_parts = []
        
        if title:
            summary_parts.append(title)
        
        if description:
            # 取描述的前100字符
            desc = description.strip()
            if len(desc) > 100:
                # 尝试在句号处截断
                sentences = desc.split('.')
                desc_parts = []
                total_length = 0
                for sent in sentences:
                    if total_length + len(sent) < 100:
                        desc_parts.append(sent.strip())
                        total_length += len(sent) + 1
                    else:
                        break
                if desc_parts:
                    desc = ". ".join(desc_parts)
                    if desc and not desc.endswith('.'):
                        desc += "."
                    desc = desc[:100] + "..." if len(desc) > 100 else desc
            summary_parts.append(desc)
        
        # 添加来源信息
        source_info = ""
        if "github" in url.lower() or "github" in source.lower():
            source_info = "（GitHub代码仓库）"
        elif "colab" in url.lower() or "colab" in source.lower():
            source_info = "（Google Colab笔记本）"
        elif "kaggle" in url.lower() or "kaggle" in source.lower():
            source_info = "（Kaggle笔记本）"
        elif "stackoverflow" in url.lower() or "stackoverflow" in source.lower():
            source_info = "（Stack Overflow问答）"
        elif "paperswithcode" in url.lower() or "papers with code" in source.lower():
            source_info = "（Papers with Code）"
        
        if summary_parts:
            summary = ". ".join([p for p in summary_parts if p])
            if source_info:
                summary += source_info
            return summary[:200] + "..." if len(summary) > 200 else summary
        elif source_info:
            return f"代码资源{source_info}"
        else:
            return "代码资源"
    
    # 默认情况
    if all_text:
        return all_text[:150] + "..." if len(all_text) > 150 else all_text
    else:
        return "资源内容"


def compute_similarity(user_docs: List[str], resources: List[Dict], resource_type: str) -> List[Tuple[Dict, float]]:
    """
    计算用户文档与资源之间的相似度
    
    Args:
        user_docs: 用户上传的文档列表（已清洗的文本）
        resources: 资源列表，每个资源是Dict，包含content/description等字段
        resource_type: 资源类型 ('txt', 'video')
    
    Returns:
        List[Tuple[Dict, float]]: (资源, 相似度分数) 的列表，按相似度降序排列
    """
    if not user_docs or not resources:
        return []
    
    # 清洗用户文档
    cleaned_user_docs = [clean_text(doc) for doc in user_docs]
    user_text = " ".join(cleaned_user_docs)  # 合并所有用户文档
    
    # 提取资源文本内容
    resource_texts = []
    resource_objs = []
    
    for res in resources:
        # 根据资源类型提取文本
        if resource_type == "txt":
            text = res.get("content", res.get("title", ""))
        elif resource_type == "video":
            text = res.get("description", res.get("title", ""))
        elif resource_type == "code":
            text = res.get("description", res.get("title", ""))
        else:
            text = str(res.get("title", ""))
        
        if text:
            resource_texts.append(clean_text(text))
            resource_objs.append(res)
    
    if not resource_texts:
        return []
    
    # 使用TF-IDF向量化
    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=1,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
            norm="l2"
        )
        
        # 合并用户文档和资源文本
        all_texts = [user_text] + resource_texts
        vectors = vectorizer.fit_transform(all_texts)
        
        # 用户文档向量（第一个）
        user_vector = vectors[0:1]
        # 资源向量（其余）
        resource_vectors = vectors[1:]
        
        # 计算余弦相似度
        similarities = cosine_similarity(user_vector, resource_vectors)[0]
        
        # 组合结果并排序
        results = list(zip(resource_objs, similarities))
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    except Exception as e:
        print(f"Error computing similarity: {e}")
        # 如果向量化失败，返回所有资源（相似度为0）
        return [(res, 0.0) for res in resource_objs]


def recommend_best_resources(
    user_folder_path: str,
    all_resources: Dict[str, List[Dict]],
    top_k_per_type: int = 5
) -> Dict[str, List[Dict]]:
    """
    使用CBF推荐系统筛选最佳资源
    
    Args:
        user_folder_path: 用户上传的文件夹路径
        all_resources: {
            "txt": [...],
            "video": [...],
        }
        top_k_per_type: 每种类型选择前K个
    
    Returns:
        筛选后的资源字典，格式同all_resources
    """
    # 读取用户文档
    user_docs = read_txt_files(user_folder_path)
    
    if not user_docs:
        print("Warning: No user documents found, returning all resources")
        return all_resources
    
    recommended = {}
    
    # 对每种类型的资源进行推荐
    for resource_type, resources in all_resources.items():
        if not resources:
            recommended[resource_type] = []
            continue
        
        # 计算相似度
        similarity_results = compute_similarity(user_docs, resources, resource_type)
        print(f"  [{resource_type}] 计算了 {len(similarity_results)} 个资源的相似度")
        
        if similarity_results:
            max_score = max(score for _, score in similarity_results)
            min_score = min(score for _, score in similarity_results)
            avg_score = sum(score for _, score in similarity_results) / len(similarity_results)
            print(f"  [{resource_type}] 相似度范围: {min_score:.4f} - {max_score:.4f}, 平均: {avg_score:.4f}")
        
        # 在相似度基础上叠加轻量领域加分，优先保留更像“AI学习资源”的候选。
        reranked_results = [
            (res, score, score + compute_domain_bonus(res))
            for res, score in similarity_results
        ]
        sorted_results = sorted(reranked_results, key=lambda x: x[2], reverse=True)
        print(f"  [{resource_type}] 相似度+领域加分排序完成，准备选择前 {top_k_per_type} 个")
        
        # 直接取前top_k个，多选一些以防被relevance过滤
        # 摘要生成不在这里做，避免在推荐阶段重复调用AI/摘要逻辑。
        # 最终展示时再按需生成一次即可。
        top_resources = []
        relevance_filtered = 0
        for res, score, final_score in sorted_results[:top_k_per_type * 2]:  # 多选一些候选
            # 额外检查：过滤掉明显不相关的资源
            if is_relevant_resource(res, user_docs):
                res["similarity_score"] = float(score)
                res["ranking_score"] = float(final_score)
                top_resources.append(res)
                if len(top_resources) >= top_k_per_type:
                    break
            else:
                relevance_filtered += 1
        
        print(f"  [{resource_type}] 相关性过滤掉: {relevance_filtered} 个, 最终推荐: {len(top_resources)} 个")
        
        recommended[resource_type] = top_resources
    
    return recommended


def save_recommended_resources(
    recommended: Dict[str, List[Dict]],
    output_folder: str
):
    """
    将推荐结果保存到文件夹
    
    Args:
        recommended: 推荐结果字典
        output_folder: 输出文件夹路径
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # 为每种类型创建子文件夹
    for resource_type, resources in recommended.items():
        type_folder = os.path.join(output_folder, resource_type)
        os.makedirs(type_folder, exist_ok=True)
        
        for i, res in enumerate(resources):
            if resource_type == "txt":
                # 清理标题
                cleaned_title = clean_title(res.get('title', 'resource'))
                filename = f"{i+1}_{sanitize_filename(cleaned_title)}.txt"
                filepath = os.path.join(type_folder, filename)
                content = res.get("content", "")
                # 清理内容：移除联系方式、部门信息等无关内容
                cleaned_content = clean_extracted_content(content)
                metadata = f"Source: {res.get('source', 'Unknown')}\n"
                metadata += f"URL: {res.get('url', '')}\n"
                metadata += f"Similarity Score: {res.get('similarity_score', 0.0):.4f}\n"
                metadata += "\n" + "="*50 + "\n\n"
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(metadata + cleaned_content)
            
            elif resource_type == "video":
                # 清理标题
                cleaned_title = clean_title(res.get('title', 'video'))
                filename = f"{i+1}_{sanitize_filename(cleaned_title)}.txt"
                filepath = os.path.join(type_folder, filename)
                content = f"Title: {cleaned_title}\n"
                content += f"URL: {res.get('url', '')}\n"
                description = res.get('description', '')
                if description:
                    content += f"Description: {description}\n"
                content += f"Similarity Score: {res.get('similarity_score', 0.0):.4f}\n"
                if res.get("thumbnail"):
                    content += f"Thumbnail: {res.get('thumbnail')}\n"
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
            
            elif resource_type == "code":
                # 清理标题
                cleaned_title = clean_title(res.get('title', 'code'))
                filename = f"{i+1}_{sanitize_filename(cleaned_title)}.txt"
                filepath = os.path.join(type_folder, filename)
                content = f"Title: {cleaned_title}\n"
                content += f"URL: {res.get('url', '')}\n"
                content += f"Source: {res.get('source', 'Unknown')}\n"
                description = res.get('description', '')
                if description:
                    content += f"Description: {description}\n"
                content += f"Similarity Score: {res.get('similarity_score', 0.0):.4f}\n"
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)


def sanitize_filename(filename: str) -> str:
    """清理文件名，移除非法字符"""
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = filename[:100]  # 限制长度
    return filename

