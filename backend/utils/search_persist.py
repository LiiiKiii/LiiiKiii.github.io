# -*- coding: utf-8 -*-
"""Persist raw multi-source search hits to disk (under data/results/)."""

import os
from typing import Any, Dict

from backend.core.resource_searcher import clean_extracted_content, clean_title
from backend.utils.file_utils import sanitize_filename


def save_search_results(all_resources: Dict[str, Any], folder_name: str, results_dir: str) -> None:
    """Write search results to ``results_dir / folder_name / {txt,video,code}/``."""
    result_folder = os.path.join(results_dir, folder_name)
    os.makedirs(result_folder, exist_ok=True)

    for resource_type, resources in all_resources.items():
        type_folder = os.path.join(result_folder, resource_type)
        os.makedirs(type_folder, exist_ok=True)

        for i, res in enumerate(resources):
            if resource_type == "txt":
                cleaned_title = clean_title(res.get("title", "resource"))
                filename = f"{i+1}_{sanitize_filename(cleaned_title)[:50]}.txt"
                filepath = os.path.join(type_folder, filename)
                content = res.get("content", "")
                cleaned_content = clean_extracted_content(content)
                metadata = f"Source: {res.get('source', 'Unknown')}\n"
                metadata += f"URL: {res.get('url', '')}\n"
                metadata += "\n" + "=" * 50 + "\n\n"
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(metadata + cleaned_content)

            elif resource_type == "video":
                cleaned_title = clean_title(res.get("title", "video"))
                filename = f"{i+1}_{sanitize_filename(cleaned_title)[:50]}.txt"
                filepath = os.path.join(type_folder, filename)
                content = f"Title: {cleaned_title}\n"
                content += f"URL: {res.get('url', '')}\n"
                description = res.get("description", "")
                if description:
                    content += f"Description: {description}\n"
                if res.get("thumbnail"):
                    content += f"Thumbnail: {res.get('thumbnail')}\n"
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

            elif resource_type == "code":
                cleaned_title = clean_title(res.get("title", "code"))
                filename = f"{i+1}_{sanitize_filename(cleaned_title)[:50]}.txt"
                filepath = os.path.join(type_folder, filename)
                content = f"Title: {cleaned_title}\n"
                content += f"URL: {res.get('url', '')}\n"
                content += f"Source: {res.get('source', 'Unknown')}\n"
                description = res.get("description", "")
                if description:
                    content += f"Description: {description}\n"
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
