#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-Pedia: AI-Embedded Multimedia Resource Recommender
Main application file

AI-embedded multimedia recommender for AI education
"""

import json
import os
import shutil
import smtplib
import threading
import time
import traceback
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from flask import Flask, Response, jsonify, render_template, request, send_file, stream_with_context
from werkzeug.utils import secure_filename

import config as cfg
from backend.core.ai_summarizer import generate_resource_summary
from backend.core.keyword_extractor import extract_keywords_from_folder
from backend.core.recommender import recommend_best_resources, save_recommended_resources
from backend.core.resource_searcher import search_all_resources
from backend.utils.file_utils import (
    cleanup_user_data,
    convert_all_pdfs_to_txt,
    count_pdf_files,
    create_output_zip,
    extract_zip,
)
from backend.utils.search_persist import save_search_results

UPLOAD_DIR = cfg.UPLOAD_DIR
RESULTS_DIR = cfg.RESULTS_DIR
OUTPUT_DIR = cfg.OUTPUT_DIR

for dir_path in (UPLOAD_DIR, RESULTS_DIR, OUTPUT_DIR):
    os.makedirs(dir_path, exist_ok=True)

app = Flask(
    __name__,
    template_folder="frontend/templates",
    static_folder="frontend/static",
)
app.config["MAX_CONTENT_LENGTH"] = cfg.MAX_UPLOAD_BYTES
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR


def _env_flag(name, default=False):
    """Parse boolean-like environment variables."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(name, default):
    """Read a string environment variable with a fallback."""
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default



@app.route("/")
def index():
    """Home page."""
    return render_template("index.html")


@app.route("/help")
def help():
    """Help page."""
    return render_template("help.html")


@app.route("/contact")
def contact_page():
    """Join us page."""
    return render_template("contact.html")


@app.route("/progress")
def progress():
    """Progress page."""
    return render_template("progress.html")


@app.route("/ai-enhance")
def ai_enhance():
    """AI enhancement page."""
    return render_template("ai-enhance.html")


@app.route("/health")
def health():
    """Container/deployment health check."""
    return jsonify({
        "status": "ok",
        "service": "ai-pedia",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })


@app.route("/contact", methods=["POST"])
def contact():
    """Handle join-us form submissions."""
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        email = data.get('email', '').strip()
        subject = data.get('subject', '').strip()
        message = data.get('message', '').strip()
        
        if not all([name, email, subject, message]):
            return jsonify({"success": False, "error": "请填写所有必填字段"}), 400
        
        if '@' not in email or '.' not in email.split('@')[1]:
            return jsonify({"success": False, "error": "请输入有效的邮箱地址"}), 400
        
        recipient_email = "czzx58@durham.ac.uk"
        
        email_body = f"""
New message received from the Join Us form:

Name: {name}
Email: {email}
Subject: {subject}

Message:
{message}

---
This message was sent automatically by the AI-Pedia Join Us form.
        """
        
        msg = MIMEMultipart()
        msg['From'] = email  # Use the user-provided email as the sender
        msg['To'] = recipient_email
        msg['Subject'] = f"Join Us: {subject}"
        msg['Reply-To'] = email  # Set the reply-to address to the user email
        
        msg.attach(MIMEText(email_body, 'plain', 'utf-8'))
        
        try:
            
            # smtp_server = "smtp.gmail.com"
            # smtp_port = 587
            # smtp_user = "your-email@gmail.com"
            # smtp_password = "your-app-password"
            # 
            # server = smtplib.SMTP(smtp_server, smtp_port)
            # server.starttls()
            # server.login(smtp_user, smtp_password)
            # server.send_message(msg)
            # server.quit()
            
            contact_log_path = os.path.join(cfg.DATA_DIR, "contact_logs.txt")
            os.makedirs(os.path.dirname(contact_log_path), exist_ok=True)
            with open(contact_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"Time: {datetime.now().isoformat()}\n")
                f.write(email_body)
                f.write(f"\n{'='*50}\n")
            
            return jsonify({
                "success": True,
                "message": "Your message has been sent successfully. We will reply as soon as possible."
            })
            
        except Exception as e:
            print(f"Error while sending email: {str(e)}")
            return jsonify({
                "success": False,
                "error": "There was an error while sending the email. Please try again later or contact czzx58@durham.ac.uk directly."
            }), 500
            
    except Exception as e:
        print(f"Error while handling the join-us form: {str(e)}")
        return jsonify({
            "success": False,
            "error": "There was an error while processing the request. Please try again later."
        }), 500


@app.route("/upload", methods=["POST"])
def upload_folder():
    """Upload a folder through a ZIP file."""
    if 'folder' not in request.files:
        return jsonify({"error": "没有上传文件"}), 400
    
    file = request.files['folder']
    if file.filename == '':
        return jsonify({"error": "文件名为空"}), 400
    
    if not file.filename.lower().endswith('.zip'):
        return jsonify({"error": "请上传zip格式的文件夹"}), 400
    
    folder_name = secure_filename(file.filename.replace('.zip', ''))
    upload_path = os.path.join(UPLOAD_DIR, folder_name)
    os.makedirs(upload_path, exist_ok=True)
    
    zip_path = os.path.join(upload_path, file.filename)
    file.save(zip_path)
    
    extract_path = os.path.join(upload_path, "extracted")
    os.makedirs(extract_path, exist_ok=True)
    
    if not extract_zip(zip_path, extract_path):
        return jsonify({"error": "解压失败"}), 400
    
    pdf_count = count_pdf_files(extract_path)
    
    original_txt_count = 0
    for root, dirs, files in os.walk(extract_path):
        for f in files:
            if f.startswith('._') or f.startswith('.DS_Store'):
                continue
            if f.lower().endswith('.txt') and not f.lower().endswith('_pdf.txt'):
                original_txt_count += 1
    
    conversion_result = {"success_count": 0, "failed_count": 0}
    
    if pdf_count > 0:
        print(f"Target files found. Starting conversion...")
        conversion_result = convert_all_pdfs_to_txt(extract_path)
        print(f"PDF conversion complete: success {conversion_result['success_count']} files, failed {conversion_result['failed_count']} files")
    
    total_valid_files = original_txt_count + conversion_result.get('success_count', 0)
    if total_valid_files < cfg.MIN_VALID_DOCUMENTS:
        shutil.rmtree(upload_path, ignore_errors=True)
        return jsonify({
            "error": f"文件夹中有效的txt/pdf文件数量不足（需要至少{cfg.MIN_VALID_DOCUMENTS}个，当前有{total_valid_files}个：{original_txt_count}个txt文件 + {conversion_result.get('success_count', 0)}个成功转换的PDF文件）"
        }), 400
    
    converted_txt = conversion_result.get('success_count', 0)
    
    return jsonify({
        "success": True,
        "folder_name": folder_name,
        "txt_count": total_valid_files,  # Total usable TXT files (original TXT plus converted PDF TXT)
        "pdf_count": pdf_count,
        "original_txt": original_txt_count,
        "converted_txt": converted_txt,
        "message": f"成功上传，包含{original_txt_count}个txt文件和{pdf_count}个pdf文件（成功转换{converted_txt}个）"
    })


def send_progress_event(progress, message, step=None, details=None):
    """Send an SSE progress event."""
    event_data = {
        "progress": progress,
        "message": message,
        "step": step,
        "details": details
    }
    return f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"


@app.route("/process", methods=["POST"])
def process_folder():
    """Process an uploaded folder and stream progress over SSE."""
    data = request.get_json()
    folder_name = data.get("folder_name")
    openai_api_key = data.get("openai_api_key")  # OpenAI API key passed from the frontend
    
    if not folder_name:
        return jsonify({"error": "缺少folder_name参数"}), 400
    
    upload_path = os.path.join(UPLOAD_DIR, folder_name, "extracted")
    if not os.path.isdir(upload_path):
        return jsonify({"error": "文件夹不存在"}), 404
    
    def generate():
        try:
            yield send_progress_event(5, "🚀 开始处理文件...", "start", "正在初始化处理流程...")
            
            yield send_progress_event(10, "📝 正在分析文档内容，提取关键词和主题...", "extract_keywords", "正在读取文档并分析内容...")
            keywords = extract_keywords_from_folder(upload_path, top_k=cfg.KEYWORD_TOP_K)
            if not keywords:
                yield send_progress_event(0, "❌ 无法提取关键词", "error", "处理失败")
                return
            
            yield send_progress_event(25, f"✅ 关键词提取完成，共提取 {len(keywords)} 个关键词", "keywords_extracted", f"关键词: {', '.join(keywords[:5])}...")
            
            yield send_progress_event(30, "🔍 开始搜索相关资源...", "search_resources", "正在搜索文本、视频和代码资源...")
            
            progress_queue = []
            
            def progress_callback(progress_info):
                """Progress callback that collects status updates."""
                progress_queue.append(progress_info)
            
            search_result = [None]
            search_error = [None]
            search_done = threading.Event()
            
            def search_thread():
                try:
                    result = search_all_resources(
                        keywords, max_per_type=cfg.SEARCH_MAX_PER_TYPE, progress_callback=progress_callback
                    )
                    search_result[0] = result
                except Exception as e:
                    search_error[0] = e
                finally:
                    search_done.set()
            
            thread = threading.Thread(target=search_thread)
            thread.daemon = True
            thread.start()
            
            while not search_done.is_set() or progress_queue:
                while progress_queue:
                    progress_info = progress_queue.pop(0)
                    progress_data = {
                        "type": "search_progress",
                        "progress": progress_info
                    }
                    yield f"data: {json.dumps(progress_data, ensure_ascii=False)}\n\n"
                
                time.sleep(0.1)
            
            thread.join(timeout=300)
            
            if search_error[0]:
                raise search_error[0]
            
            all_resources = search_result[0]
            
            txt_found = len(all_resources.get("txt", []))
            video_found = len(all_resources.get("video", []))
            code_found = len(all_resources.get("code", []))
            
            yield send_progress_event(50, f"📊 资源搜索完成", "resources_found", 
                                    f"找到 {txt_found} 个文本资源, {video_found} 个视频资源, {code_found} 个代码资源")
            
            yield send_progress_event(55, "💾 正在保存搜索结果...", "save_results", "正在保存到本地文件...")
            save_search_results(all_resources, folder_name, RESULTS_DIR)
            yield send_progress_event(60, "✅ 搜索结果已保存", "results_saved", "")
            
            yield send_progress_event(65, "🎯 开始推荐筛选...", "recommend", "正在计算相似度并筛选最佳资源...")
            recommended = recommend_best_resources(
                upload_path,
                all_resources,
                top_k_per_type=cfg.RECOMMEND_TOP_K_PER_TYPE,
            )
            
            txt_rec_count = len(recommended.get("txt", []))
            video_rec_count = len(recommended.get("video", []))
            code_rec_count = len(recommended.get("code", []))
            
            yield send_progress_event(80, f"✨ 推荐筛选完成", "recommend_done", 
                                    f"推荐了 {txt_rec_count} 个文本资源, {video_rec_count} 个视频资源, {code_rec_count} 个代码资源")
            
            yield send_progress_event(85, "💾 正在保存推荐结果...", "save_recommended", "正在保存推荐资源...")
            output_folder = os.path.join(OUTPUT_DIR, folder_name)
            save_recommended_resources(recommended, output_folder)
            yield send_progress_event(90, "✅ 推荐结果已保存", "recommended_saved", "")
            
            yield send_progress_event(95, "📦 正在准备最终数据...", "prepare_data", "正在整理数据...")
            
            stats = {
                "keywords": len(keywords),
                "txt_found": txt_found,
                "video_found": video_found,
                "code_found": code_found,
                "txt_recommended": txt_rec_count,
                "video_recommended": video_rec_count,
                "code_recommended": code_rec_count,
            }
            
            recommended_resources = {}
            for resource_type, resources in recommended.items():
                recommended_resources[resource_type] = []
                for res in resources:
                    resource_data = {
                        "title": res.get("title", "无标题"),
                        "url": res.get("url", ""),
                        "source": res.get("source", "Unknown"),
                        "similarity_score": res.get("similarity_score", 0.0),
                    }
                    
                    existing_summary = res.get("summary")
                    existing_summary_type = res.get("summary_type")
                    should_regenerate_summary = bool(openai_api_key)

                    if existing_summary and not should_regenerate_summary:
                        resource_data["summary"] = existing_summary
                        resource_data["summary_type"] = existing_summary_type or "cached"
                    else:
                        summary_result = generate_resource_summary(res, resource_type, openai_api_key=openai_api_key)
                        if summary_result and summary_result.get("summary"):
                            resource_data["summary"] = summary_result["summary"]
                            resource_data["summary_type"] = summary_result.get("summary_type", "ai_generated")
                            res["summary"] = resource_data["summary"]
                            res["summary_type"] = resource_data["summary_type"]
                        else:
                            resource_data["summary"] = existing_summary if existing_summary else None
                            resource_data["summary_type"] = existing_summary_type if existing_summary else None
                    
                    if resource_type == "txt":
                        content = res.get("content", "")
                        if content:
                            resource_data["description"] = content[:200] + "..." if len(content) > 200 else content
                    elif resource_type == "video":
                        if res.get("description"):
                            resource_data["description"] = res.get("description")
                        if res.get("thumbnail"):
                            resource_data["thumbnail"] = res.get("thumbnail")
                    elif resource_type == "code":
                        if res.get("description"):
                            resource_data["description"] = res.get("description")
                    
                    recommended_resources[resource_type].append(resource_data)
            
            final_data = {
                "progress": 100,
                "message": "✨ 处理完成！",
                "step": "complete",
                "success": True,
                "keywords": keywords,
                "stats": stats,
                "recommended_resources": recommended_resources
            }
            yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
            
            time.sleep(0.5)
            cleanup_folder = os.path.join(UPLOAD_DIR, folder_name)
            if os.path.exists(cleanup_folder):
                try:
                    shutil.rmtree(cleanup_folder)
                except OSError:
                    pass

        except Exception as e:
            error_data = {
                "progress": 0,
                "message": f"❌ 处理失败: {str(e)}",
                "step": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route("/download/<folder_name>")
def download_output(folder_name):
    """Download the recommended ZIP file and clean up user data afterwards."""
    output_folder = os.path.join(OUTPUT_DIR, folder_name)
    zip_path = os.path.join(OUTPUT_DIR, f"{folder_name}_recommended.zip")
    
    if not os.path.isdir(output_folder):
        return jsonify({"error": "输出文件夹不存在"}), 404
    
    if not os.path.isfile(zip_path):
        if not create_output_zip(output_folder, zip_path):
            return jsonify({"error": "创建zip文件失败"}), 500
    
    response = send_file(
        zip_path,
        as_attachment=True,
        download_name=f"{folder_name}_recommended.zip",
        mimetype="application/zip"
    )
    
    def cleanup_after_download():
        time.sleep(2)  # Wait 2 seconds to make sure the download has started
        cleanup_result = cleanup_user_data(folder_name, cfg.PROJECT_ROOT)
        print(f"Cleaning up user data {folder_name}: {cleanup_result['message']}")
    
    cleanup_thread = threading.Thread(target=cleanup_after_download)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    return response


@app.route("/status/<folder_name>")
def get_status(folder_name):
    """Get processing status."""
    result_folder = os.path.join(RESULTS_DIR, folder_name)
    output_folder = os.path.join(OUTPUT_DIR, folder_name)
    
    return jsonify({
        "folder_name": folder_name,
        "has_results": os.path.isdir(result_folder),
        "has_output": os.path.isdir(output_folder),
        "ready_for_download": os.path.isdir(output_folder)
    })


@app.route("/cleanup/<folder_name>", methods=["POST"])
def cleanup_data(folder_name):
    """Manually clean up user data."""
    try:
        cleanup_result = cleanup_user_data(folder_name, cfg.PROJECT_ROOT)
        if cleanup_result["success"]:
            return jsonify({
                "success": True,
                "message": cleanup_result["message"],
                "deleted": cleanup_result["deleted"]
            })
        else:
            return jsonify({
                "success": False,
                "message": cleanup_result["message"]
            }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"清理失败: {str(e)}"
        }), 500


if __name__ == "__main__":
    print("=" * 50)
    print("AI-Pedia")
    print("=" * 50)
    
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    if openai_key:
        masked_key = openai_key[:10] + "..." + openai_key[-4:] if len(openai_key) > 14 else "***"
        print(f"✓ OpenAI API Key: {masked_key} (configured)")
    else:
        print("⚠ Warning: OPENAI_API_KEY was not detected")
        print("  Note: set the environment variable to enable optional LLM summaries")
        print("  Example: export OPENAI_API_KEY='your-key-here'")
    
    print("=" * 50)
    print(f"Upload directory: {UPLOAD_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 50)
    host = _env_str("APP_HOST", "127.0.0.1")
    port = int(_env_str("APP_PORT", "5000"))
    access_host = "localhost" if host in {"127.0.0.1", "0.0.0.0"} else host
    print(f"Open: http://{access_host}:{port}")
    print("Press Ctrl+C to stop the service")
    print("=" * 50)
    flask_env = os.getenv("FLASK_ENV", "").strip().lower()
    debug_mode = _env_flag("FLASK_DEBUG", default=(flask_env == "development"))
    app.run(host=host, port=port, debug=debug_mode)
