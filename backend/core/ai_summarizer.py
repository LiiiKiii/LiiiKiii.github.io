#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summary generation utilities for AI-Pedia."""

import os
import re
from typing import Dict, Optional

# Try importing OpenAI client
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Fallback import for other SDKs
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


OPENAI_SUMMARY_MODEL = os.environ.get("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")


def get_openai_api_key(api_key_from_request: Optional[str] = None) -> Optional[str]:
    """Get openai api key."""
    # Prefer key from request
    if api_key_from_request:
        return api_key_from_request
    
    # Else read from environment
    return os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")


def generate_summary_with_openai(content: str, resource_type: str, title: str = "", max_tokens: int = 150, api_key: Optional[str] = None) -> Optional[str]:
    """Handle generate summary with openai."""
    if not HAS_OPENAI:
        return None
    
    # Resolve API key (request overrides env)
    api_key = api_key or get_openai_api_key()
    if not api_key:
        return None
    
    try:
            # Build prompt for resource type/length
        if resource_type == "txt":
            prompt = f"""Write a concise 3-4 sentence summary for the academic resource below.

Cover:
1. What it is about.
2. What stands out.
3. What the learner can gain.

Requirements:
- Keep the summary direct and informative.
- Do not repeat the title.
- Avoid vague phrases such as "related to your learning materials".

Title: {title}

Content snippet:
{content}

Summary:"""
        elif resource_type == "video":
            prompt = f"""Write a concise 2-3 sentence summary for the video resource below.

Requirements:
1. Identify the type and source of the video.
2. Explain clearly what the video teaches or covers.
3. Add one useful detail or use case if needed.
4. Do not repeat the title or use vague phrasing.

Title: {title}

Description: {content}

Summary:"""
        elif resource_type == "code":
            prompt = f"""Write a concise 2-3 sentence summary for the code resource below.

Requirements:
1. Identify the type and source of the repository.
2. Explain what it implements, which technology it uses, or which problem it solves.
3. Add one useful feature or application scenario if needed.
4. Do not repeat the title or use vague phrasing.

Title: {title}

Description: {content}

Summary:"""
        
        # OpenAI Python SDK client
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=OPENAI_SUMMARY_MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional academic resource assistant. Write concise summaries that cover: 1) what the resource is about, 2) what stands out, and 3) what the learner can gain. Keep the writing direct and avoid vague phrasing."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                timeout=10
            )
            summary = response.choices[0].message.content.strip()
        except (ImportError, AttributeError):
            # Legacy openai package
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model=OPENAI_SUMMARY_MODEL,
                messages=[
                    {"role": "system", "content": "You are a professional academic resource assistant. Write concise summaries that cover: 1) what the resource is about, 2) what stands out, and 3) what the learner can gain. Keep the writing direct and avoid vague phrasing."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                timeout=10
            )
            summary = response.choices[0].message.content.strip()
        
        # Trim quotes/extra spaces from summary
        summary = re.sub(r'^["\']|["\']$', '', summary)
        summary = re.sub(r'\s+', ' ', summary).strip()
        
        return summary if summary else None
        
    except Exception as e:
        return None


def generate_summary_with_fallback(content: str, resource_type: str, title: str = "") -> str:
    """Handle generate summary with fallback."""
    # Strip noise from body
    content = re.sub(r'<[^>]+>', '', content)  # strip HTML tags
    content = re.sub(r'\s+', ' ', content).strip()
    
    if resource_type == "txt":
        # Pick 2–3 intro sentences
        sentences = re.split(r'[.!?]\s+', content)
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
        
        if meaningful_sentences:
            # First sentences; skip title/meta echo
            summary_parts = []
            for sent in meaningful_sentences[:3]:
                # Skip sentences too close to title
                if title and title.lower() in sent.lower()[:50]:
                    continue
                if any(phrase in sent for phrase in ["From Wikipedia", "Redirected from", "Part of a series"]):
                    continue
                # Skip very short lines
                if len(sent) < 40:
                    continue
                summary_parts.append(sent)
                # Cap ~3 sentences / 120 chars
                if len(summary_parts) >= 3 or len('. '.join(summary_parts)) > 120:
                    break
            
            if summary_parts:
                summary = '. '.join(summary_parts)
                if not summary.endswith('.'):
                    summary += '.'
                # Enforce 120-char summary cap
                if len(summary) > 120:
                    summary = summary[:117] + "..."
                return summary
        
        # Fallback: strip meta then excerpt
        if content:
            # Remove title and boilerplate
            content_without_title = content
            if title:
                content_without_title = content.replace(title, '')
            # Drop leading metadata lines
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
            # Else first 120 chars of raw text
            return content_without_title[:120] + "..." if len(content_without_title) > 120 else content_without_title
    
    elif resource_type == "video":
        if content:
            # Video descriptions are often short
            desc = content[:120]
            if len(content) > 120:
                # Clip at sentence end
                sentences = desc.split('.')
                if len(sentences) > 1:
                    desc = '. '.join(sentences[:-1]) + '.'
            return desc
        return title if title else "Video resource"
    
    elif resource_type == "code":
        if content:
            # Code repo blurb
            desc = content[:100]
            return desc + "..." if len(content) > 100 else desc
        return title if title else "Code resource"
    


def extract_abstract_from_content(content: str) -> str:
    """Extract abstract from content."""
    if not content:
        return ""
    
    # Support formats:
    # 2. "Abstract: {abstract}\n\n" (English)
    patterns = [
        r'摘要[：:]\s*(.+?)(?:\n\n|论文链接|arXiv链接|请访问|$)',  # Non-greedy to delimiter
        r'Abstract[：:]\s*(.+?)(?:\n\n|论文链接|arXiv链接|请访问|$)',  # English format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            abstract = match.group(1).strip()
            # Normalize abstract whitespace
            abstract = re.sub(r'\n+', ' ', abstract)  # newlines to spaces
            abstract = re.sub(r'\s+', ' ', abstract)  # collapse spaces
            # strip author line if any
            abstract = re.sub(r'作者[：:][^。]+。', '', abstract)
            abstract = re.sub(r'Authors?[：:][^.]+\.[\s.]*', '', abstract, flags=re.IGNORECASE)
            abstract = abstract.strip()
            # require min abstract length
            if abstract and len(abstract) > 30:  # min 30 chars
                return abstract
    
    return ""


def generate_simple_wikipedia_summary(content: str, title: str) -> str:
    """Handle generate simple wikipedia summary."""
    # one-line blurb only
    return f"这是关于{title}的百科文章。"


def generate_resource_summary(resource: Dict, resource_type: str, openai_api_key: Optional[str] = None) -> Dict:
    """Handle generate resource summary."""
    title = resource.get("title", "")
    content = resource.get("content", "")
    description = resource.get("description", "")
    url = resource.get("url", "")
    source = resource.get("source", "")
    
    # Text: if arXiv with abstract
    if resource_type == "txt":
        abstract = extract_abstract_from_content(content)
        if abstract:
            # Heuristic arXiv source check
            is_arxiv = any(keyword in (source or "").lower() for keyword in ["arxiv", "google scholar", "scholar"])
            is_arxiv = is_arxiv or "arxiv" in (url or "").lower()
            
            if is_arxiv:
                # Return abstract; UI can expand
                return {
                    "summary": abstract,
                    "summary_type": "abstract"
                }
    
    # Assemble text for model
    content_text = ""
    if resource_type == "txt":
        content_text = content or description or ""
    elif resource_type == "video":
        content_text = description or content or ""
    elif resource_type == "code":
        content_text = description or content or ""
    
    # Wikipedia: more context for LLM
    # truncate for token limits
    max_content_length = 2000 if "Wikipedia" in title or len(content_text) > 3000 else 1000
    if len(content_text) > max_content_length:
        content_text = content_text[:max_content_length] + "..."
    
    # Call OpenAI for summary
    # ~100 tokens, structured blurb
    max_tokens = 100
    ai_summary = generate_summary_with_openai(content_text, resource_type, title, max_tokens=max_tokens, api_key=openai_api_key)
    if ai_summary:
        return {
            "summary": ai_summary,
            "summary_type": "ai_generated"
        }
    
    # On API failure, branch by type
    if resource_type == "txt":
        # Wikipedia branch
        is_wikipedia = "Wikipedia" in (title or "") or "wikipedia" in (url or "").lower() or "wikipedia" in (source or "").lower()
        
        if is_wikipedia:
            # Wikipedia: short line
            simple_summary = generate_simple_wikipedia_summary(content_text, title)
            return {
                "summary": simple_summary,
                "summary_type": "wikipedia_simple"
            }
        else:
            # Other text: no fallback
            return {
                "summary": None,
                "summary_type": None
            }
    else:
        # Video/code: return None on failure
        return {
            "summary": None,
            "summary_type": None
        }
