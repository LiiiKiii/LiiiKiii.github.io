// AI多媒体推荐系统 - 主JavaScript文件

let currentFolderName = null;
let processingInterval = null;

// DOM元素
const folderInput = document.getElementById('folder-input');
const uploadArea = document.getElementById('upload-area');
const uploadBtn = document.getElementById('upload-btn');
const uploadStatus = document.getElementById('upload-status');
const processingSection = document.getElementById('processing-section');
const resultsSection = document.getElementById('results-section');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const processingDetails = document.getElementById('processing-details');
// 右侧面板元素
const rightPanel = document.getElementById('right-panel');
const progressPanel = document.getElementById('progress-panel');
const resultsPanel = document.getElementById('results-panel');
const terminalOutput = document.getElementById('terminal-output');

// 视图切换相关
let isResourcesOnlyView = false;
let viewToggleFloat = document.getElementById('view-toggle-float');
let viewToggleBtn = document.getElementById('view-toggle-btn');
let viewToggleText = document.getElementById('view-toggle-text');
const mainContainer = document.getElementById('main-container');

// 初始化
document.addEventListener('DOMContentLoaded', () => {
  // 首先加载偏好设置，确保主题在页面渲染前就应用
  loadPreferences();
  triggerPageAnimation();
  setupUploadArea();
  setupUploadButton();
  setupNavigation();
  setupViewToggle();
});

// 触发页面进入动画
function triggerPageAnimation() {
  // 确保容器可见
  const container = document.getElementById('main-container');
  if (container) {
    container.style.opacity = '1';
  }
  
  // 为所有需要动画的元素重置动画（如果需要重新触发）
  const header = document.querySelector('header');
  const cards = document.querySelectorAll('.step-card');
  
  [header, ...cards].forEach(el => {
    if (el) {
      el.style.animation = 'none';
      // 强制重排
      void el.offsetHeight;
      el.style.animation = null;
    }
  });
}

// 设置上传区域
function setupUploadArea() {
  const uploadBox = uploadArea.querySelector('.upload-box');
  
  // 点击上传
  uploadBox.addEventListener('click', () => {
    folderInput.click();
  });
  
  // 文件选择
  folderInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
      handleFileSelect(file);
    }
  });
  
  // 拖拽上传
  uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#764ba2';
  });
  
  uploadBox.addEventListener('dragleave', (e) => {
    e.preventDefault();
    // 恢复为CSS中定义的默认样式，不再强制改成纯白
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = '';
  });
  
  uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    // 上传结束后同样恢复为默认样式
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = '';
    
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.zip')) {
      folderInput.files = e.dataTransfer.files;
      handleFileSelect(file);
    } else {
      const lang = getCurrentLanguage();
      const invalidFormatText = lang === 'en-US' ? I18N_MAP['en-US']['home.upload.invalidFormat'] : I18N_MAP['zh-CN']['home.upload.invalidFormat'];
      showStatus(invalidFormatText, 'error');
    }
  });
}

function handleFileSelect(file) {
  if (!file.name.endsWith('.zip')) {
    const lang = getCurrentLanguage();
    const invalidFormatText = lang === 'en-US' ? I18N_MAP['en-US']['home.upload.invalidFormat'] : I18N_MAP['zh-CN']['home.upload.invalidFormat'];
    showStatus(invalidFormatText, 'error');
    return;
  }
  
  uploadBtn.disabled = false;
  const lang = getCurrentLanguage();
  const selectedText = lang === 'en-US' ? I18N_MAP['en-US']['home.file.selected'] : I18N_MAP['zh-CN']['home.file.selected'];
  showStatus(`${selectedText} ${file.name} (${formatFileSize(file.size)})`, 'info');
}

// 设置上传按钮
function setupUploadButton() {
  uploadBtn.addEventListener('click', async () => {
    const file = folderInput.files[0];
    if (!file) {
      const lang = getCurrentLanguage();
      const noFileText = lang === 'en-US' ? I18N_MAP['en-US']['home.upload.noFileSelected'] : I18N_MAP['zh-CN']['home.upload.noFileSelected'];
      showStatus(noFileText, 'error');
      return;
    }
    
    await uploadFile(file);
  });
}

// 上传文件
async function uploadFile(file) {
  uploadBtn.disabled = true;
  const uploadingText = getCurrentLanguage() === 'en-US' ? I18N_MAP['en-US']['home.upload.uploading'] : I18N_MAP['zh-CN']['home.upload.uploading'];
  showStatus(uploadingText, 'info');
  
  const formData = new FormData();
  formData.append('folder', file);
  
  try {
    const response = await fetch('/upload', {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    
    if (response.ok && data.success) {
      currentFolderName = data.folder_name;
      // 格式化上传成功消息
      const lang = getCurrentLanguage();
      let message = data.message;
      if (lang === 'en-US') {
        // 将中文消息转换为英文
        const txtCount = data.original_txt || 0;
        const pdfCount = data.pdf_count || 0;
        const convertedCount = data.converted_txt || 0;
        message = `Successfully uploaded, including ${txtCount} txt file${txtCount !== 1 ? 's' : ''} and ${pdfCount} pdf file${pdfCount !== 1 ? 's' : ''} (${convertedCount} successfully converted)`;
      }
      showStatus(message, 'success');
      const successText = lang === 'en-US' ? I18N_MAP['en-US']['home.upload.success'] : I18N_MAP['zh-CN']['home.upload.success'];
      uploadBtn.textContent = successText;
      
      // 开始处理
      setTimeout(() => {
        startProcessing(data.folder_name);
      }, 1000);
    } else {
      const lang = getCurrentLanguage();
      const failedText = lang === 'en-US' ? I18N_MAP['en-US']['home.upload.failed'] : I18N_MAP['zh-CN']['home.upload.failed'];
      showStatus(data.error || failedText, 'error');
      uploadBtn.disabled = false;
    }
  } catch (error) {
    const lang = getCurrentLanguage();
    const failedText = lang === 'en-US' ? I18N_MAP['en-US']['home.upload.failed'] : I18N_MAP['zh-CN']['home.upload.failed'];
    showStatus(`${failedText}: ${error.message}`, 'error');
    uploadBtn.disabled = false;
  }
}

// 开始处理（使用SSE实时进度）
async function startProcessing(folderName) {
  // 显示右侧面板（单列布局）
  const container = document.getElementById('main-container');
  container.classList.add('has-results');
  
  // 显示进度窗口，隐藏结果窗口
  progressPanel.style.display = 'block';
  resultsPanel.style.display = 'none';
  
  processingSection.style.display = 'block';
  resultsSection.style.display = 'none';
  updateProgress(0, '准备开始...');
  
  // 清空终端输出
  terminalOutput.innerHTML = '';
  const startText = getCurrentLanguage() === 'en-US' ? I18N_MAP['en-US']['home.processing.start'] : I18N_MAP['zh-CN']['home.processing.start'];
  addTerminalLine(startText, 'info');
  
  // 使用fetch + ReadableStream接收SSE事件
  try {
    // 获取用户保存的OpenAI API key（如果存在）
    const openaiKey = window.getAPIKey ? window.getAPIKey('openai') : null;
    
    // 构建请求体，包含API key（如果存在）
    const requestBody = { folder_name: folderName };
    if (openaiKey) {
      requestBody.openai_api_key = openaiKey;
    }
    
    const response = await fetch('/process', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream'
      },
      body: JSON.stringify(requestBody)
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }
    
    if (!response.body) {
      throw new Error('响应体为空');
    }
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finalData = null;
    let currentStep = null;
    
    while (true) {
      const { done, value } = await reader.read();
      
      if (done) {
        break;
      }
      
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // 保留最后一个不完整的行
      
      for (const line of lines) {
        if (line.trim() === '') continue;
        
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            
            // 处理搜索进度（实时显示当前搜索的关键词）
            if (data.type === 'search_progress') {
              const progress = data.progress;
              
              if (progress.type === 'keyword_start') {
                const searchingText = getCurrentLanguage() === 'en-US' ? I18N_MAP['en-US']['home.processing.searching'] : I18N_MAP['zh-CN']['home.processing.searching'];
                addTerminalLine(`${searchingText} ${progress.index}/${progress.total}: "${progress.keyword}"`, 'info');
              }
            }
            // 处理整体进度
            else {
              currentStep = data.step || currentStep;
              
              // 更新进度
              if (data.progress !== undefined) {
                updateProgress(data.progress, data.message || '处理中...', currentStep);
              }
              
              // 更新终端输出（只显示特定步骤）
              if (data.message) {
                // 只显示初始化、分析关键词和搜索资源这三个步骤
                if (data.step in ['start', 'extract_keywords', 'search_resources']) {
                  const lineType = data.step === 'error' ? 'error' : 
                                 data.step === 'complete' ? 'success' : 'info';
                  
                  // 对于start步骤，只显示details（初始化处理流程）
                  if (data.step === 'start' && data.details) {
                    addTerminalLine(data.details, lineType);
                  }
                  // 对于extract_keywords步骤，只显示details（分析关键词）
                  else if (data.step === 'extract_keywords' && data.details) {
                    addTerminalLine(data.details, lineType);
                  }
                  // 对于search_resources步骤，显示message（开始搜索相关资源）
                  else if (data.step === 'search_resources') {
                    let message = data.message;
                    const lang = getCurrentLanguage();
                    // 翻译"开始搜索相关资源..."
                    if (message && message.includes('开始搜索相关资源')) {
                      message = lang === 'en-US' ? I18N_MAP['en-US']['home.processing.startSearching'] : message;
                    }
                    addTerminalLine(message, lineType);
                  }
                }
              }
              
              // 更新处理详情（会自动翻译）
              if (data.details) {
                updateProcessingDetails(data.details);
              }
              
              // 如果message包含需要翻译的内容，也更新处理详情
              if (data.message && (data.message.includes('开始搜索相关资源') || data.message.includes('正在搜索文本、视频和代码资源'))) {
                updateProcessingDetails(data.message);
              }
              
              // 如果是完成或错误，保存最终数据
              if (data.step === 'complete' && data.success) {
                finalData = data;
              } else if (data.step === 'error') {
                updateProgress(0, '处理失败', 'error');
                updateProcessingDetails(data.error || '处理过程中出现错误');
                showStatus(data.error || '处理失败', 'error');
                return;
              }
            }
          } catch (e) {
            console.error('解析SSE数据失败:', e, line);
          }
        }
      }
    }
    
    // 处理完成，显示结果
    if (finalData && finalData.success) {
      const completeText = getCurrentLanguage() === 'en-US' ? I18N_MAP['en-US']['home.processing.complete'] : I18N_MAP['zh-CN']['home.processing.complete'];
      const allCompleteText = getCurrentLanguage() === 'en-US' ? I18N_MAP['en-US']['home.processing.allComplete'] : I18N_MAP['zh-CN']['home.processing.allComplete'];
      updateProgress(100, completeText.replace('✨ ', ''), 'complete');
      addTerminalLine(completeText, 'success');
      updateProcessingDetails(allCompleteText);
      
      // 显示结果
      setTimeout(() => {
        showResults(finalData);
      }, 500);
    } else if (!finalData) {
      throw new Error('未收到完成数据');
    }
  } catch (error) {
    updateProgress(0, '处理失败', 'error');
    addTerminalLine('❌ 错误: ' + error.message, 'error');
    updateProcessingDetails('错误: ' + error.message);
    showStatus('处理失败: ' + error.message, 'error');
  }
}

// 添加终端输出行
function addTerminalLine(text, type = 'info') {
  const line = document.createElement('div');
  line.className = `terminal-line ${type}`;
  line.textContent = text;
  terminalOutput.appendChild(line);
  // 自动滚动到底部
  terminalOutput.scrollTop = terminalOutput.scrollHeight;
}

// 显示结果
function showResults(data) {
  processingSection.style.display = 'none';
  resultsSection.style.display = 'block';
  
  // 切换到结果面板
  progressPanel.style.display = 'none';
  resultsPanel.style.display = 'block';
  
  // 显示视图切换浮窗
  if (viewToggleFloat) {
    viewToggleFloat.style.display = 'block';
  }
  
  // 显示关键词
  const keywordsDisplay = document.getElementById('keywords-display');
  keywordsDisplay.innerHTML = '';
  if (data.keywords && data.keywords.length > 0) {
    data.keywords.forEach(keyword => {
      const tag = document.createElement('span');
      tag.className = 'keyword-tag';
      tag.textContent = keyword;
      keywordsDisplay.appendChild(tag);
    });
  }
  
  // 显示统计（找到的资源数量）
  if (data.stats) {
    console.log('接收到的stats数据:', data.stats);
    document.getElementById('txt-found').textContent = data.stats.txt_found || 0;
    document.getElementById('video-found').textContent = data.stats.video_found || 0;
    document.getElementById('code-found').textContent = data.stats.code_found || 0;
    // 推荐的资源数量会在displayResources中根据滑块值动态更新
  }
  
  // 显示推荐资源
  if (data.recommended_resources) {
    console.log('接收到的recommended_resources数据:', data.recommended_resources);
    displayResources(data.recommended_resources);
  } else {
    console.warn('没有收到recommended_resources数据');
  }
}

// 存储所有推荐资源（用于动态调整数量）
let allRecommendedResources = null;

// 显示推荐资源
function displayResources(resources) {
  // 保存所有资源
  allRecommendedResources = resources;
  
  // 根据实际资源数量动态设置滑块最大值并初始化显示
  initResourceSliders(resources);
  
  // 根据当前滑块值显示资源
  updateResourceDisplay();
}

// 初始化资源滑块（根据实际资源数量设置最大值）
function initResourceSliders(resources) {
  const resourceTypes = [
    { type: 'txt', sliderId: 'txt-count-slider', valueId: 'txt-count-value' },
    { type: 'video', sliderId: 'video-count-slider', valueId: 'video-count-value' },
    { type: 'code', sliderId: 'code-count-slider', valueId: 'code-count-value' }
  ];
  
  resourceTypes.forEach(({ type, sliderId, valueId }) => {
    const slider = document.getElementById(sliderId);
    const valueSpan = document.getElementById(valueId);
    const resourceList = resources[type] || [];
    const maxCount = Math.max(1, resourceList.length); // 至少为1
    
    if (slider) {
      // 设置最大值
      slider.max = maxCount;
      
      // 如果当前值超过最大值，调整为最大值
      const currentValue = parseInt(slider.value) || 5;
      if (currentValue > maxCount) {
        slider.value = maxCount;
      }
      
      // 设置初始值（如果资源数量少于5，使用实际数量）
      if (maxCount < 5) {
        slider.value = maxCount;
      }
      
      // 更新显示值
      valueSpan.textContent = slider.value;
      
      // 添加事件监听
      slider.addEventListener('input', (e) => {
        const count = parseInt(e.target.value);
        valueSpan.textContent = count;
        updateResourceDisplay();
      });
    }
  });
}

// 根据当前滑块值更新资源显示
function updateResourceDisplay() {
  if (!allRecommendedResources) return;
  
  // 获取每种类型的显示数量
  const txtCount = parseInt(document.getElementById('txt-count-slider')?.value) || 5;
  const videoCount = parseInt(document.getElementById('video-count-slider')?.value) || 5;
  const codeCount = parseInt(document.getElementById('code-count-slider')?.value) || 5;
  
  // 筛选每种类型的资源
  const filteredResources = {
    txt: (allRecommendedResources.txt || []).slice(0, txtCount),
    video: (allRecommendedResources.video || []).slice(0, videoCount),
    code: (allRecommendedResources.code || []).slice(0, codeCount)
  };
  
  // 更新统计数字
  document.getElementById('txt-recommended').textContent = filteredResources.txt.length;
  document.getElementById('video-recommended').textContent = filteredResources.video.length;
  document.getElementById('code-recommended').textContent = filteredResources.code.length;
  
  // 显示资源列表
  displayResourceList('txt', filteredResources.txt, 'txt-list', 'txt-count');
  displayResourceList('video', filteredResources.video, 'video-list', 'video-count');
  displayResourceList('code', filteredResources.code, 'code-list', 'code-count');
}

// 显示资源列表
function displayResourceList(type, resources, listId, countId) {
  const listElement = document.getElementById(listId);
  const countElement = document.getElementById(countId);
  
  listElement.innerHTML = '';
  countElement.textContent = resources.length;
  
  if (resources.length === 0) {
    const emptyMsg = document.createElement('div');
    emptyMsg.className = 'resource-item';
    emptyMsg.style.textAlign = 'center';
    emptyMsg.style.color = '#b0b0b0';
    const emptyText = getCurrentLanguage() === 'en-US' ? I18N_MAP['en-US']['home.resources.empty'] : I18N_MAP['zh-CN']['home.resources.empty'];
    emptyMsg.textContent = emptyText;
    listElement.appendChild(emptyMsg);
      return;
  }
  
  resources.forEach((resource, index) => {
    const item = document.createElement('div');
    item.className = 'resource-item';
    
    // 内容区域
    const contentArea = document.createElement('div');
    contentArea.className = 'resource-content';
    
    // 1. 标题
    const title = document.createElement('div');
    title.className = 'resource-item-title';
    title.textContent = `${index + 1}. ${resource.title || '无标题'}`;
    contentArea.appendChild(title);
    
    // 2. 来源和相似度（一行显示）
    const meta = document.createElement('div');
    meta.className = 'resource-item-meta';
    
    if (resource.source) {
      const sourceSpan = document.createElement('span');
      sourceSpan.className = 'resource-item-source';
      const sourceLabel = getCurrentLanguage() === 'en-US' ? I18N_MAP['en-US']['home.resources.source'] : I18N_MAP['zh-CN']['home.resources.source'];
      sourceSpan.innerHTML = `<span>${sourceLabel}</span> <strong>${resource.source}</strong>`;
      meta.appendChild(sourceSpan);
    }
    
    if (resource.similarity_score !== undefined) {
      const similaritySpan = document.createElement('span');
      similaritySpan.className = 'resource-item-similarity';
      const similarityLabel = getCurrentLanguage() === 'en-US' ? I18N_MAP['en-US']['home.resources.similarity'] : I18N_MAP['zh-CN']['home.resources.similarity'];
      similaritySpan.textContent = `${similarityLabel} ${(resource.similarity_score * 100).toFixed(1)}%`;
      meta.appendChild(similaritySpan);
    }
    
    contentArea.appendChild(meta);
    
    // 3. 简介（根据summary_type显示不同格式）
    if (resource.summary) {
      const summaryContainer = document.createElement('div');
      summaryContainer.className = 'resource-item-summary-container';
      
      const summary = document.createElement('div');
      summary.className = 'resource-item-summary';
      
      const summaryType = resource.summary_type || 'ai_generated';
      
      if (summaryType === 'abstract') {
        // Abstract类型：最多显示3行，然后显示More按钮
        const fullText = resource.summary;
        
        summary.className = 'resource-item-summary summary-has-abstract';
        summary.style.position = 'relative';
        
        // 创建文本容器
        const textWrapper = document.createElement('div');
        textWrapper.className = 'summary-abstract-text-wrapper';
        textWrapper.textContent = fullText;
        summary.appendChild(textWrapper);
        
        // 检查是否需要More按钮并添加
        const checkAndAddButton = () => {
          // 先测量完整高度
          textWrapper.style.maxHeight = 'none';
          textWrapper.style.overflow = 'visible';
          const fullHeight = textWrapper.scrollHeight;
          
          // 计算3行的高度 (line-height 1.7)
          const lineHeight = parseFloat(getComputedStyle(textWrapper).lineHeight);
          const maxHeight = lineHeight * 3;
          
          if (fullHeight > maxHeight) {
            // 设置折叠状态
            textWrapper.classList.add('summary-abstract-collapsed');
            textWrapper.style.maxHeight = maxHeight + 'px';
            textWrapper.style.overflow = 'hidden';
            
            // 添加More按钮（放在wrapper外面，这样不会被截断）
            const moreBtn = document.createElement('button');
            moreBtn.className = 'summary-more-btn more-btn-overlay';
            moreBtn.textContent = '...More';
            moreBtn.style.position = 'absolute';
            moreBtn.style.bottom = '12px';
            moreBtn.style.right = '12px';
            moreBtn.style.padding = '2px 8px';
            moreBtn.style.zIndex = '10';
            
            summary.appendChild(moreBtn);
            
            // 检查原始内容是否被截断（检查是否以省略号结尾）
            const originalText = fullText.trim();
            const hasTruncation = originalText.endsWith('...') || 
                                  originalText.endsWith('…') ||
                                  (originalText.length > 0 && originalText[originalText.length - 1] !== '.' && 
                                   originalText[originalText.length - 1] !== '!' && 
                                   originalText[originalText.length - 1] !== '?');
            
            // 如果原始内容可能被截断，添加提示
            let seeMoreHint = null;
            if (hasTruncation) {
              seeMoreHint = document.createElement('span');
              seeMoreHint.className = 'summary-see-more-hint';
              seeMoreHint.textContent = ' (see more in the article)';
              seeMoreHint.style.color = '#999';
              seeMoreHint.style.fontSize = '0.85rem';
              seeMoreHint.style.fontStyle = 'italic';
              seeMoreHint.style.marginLeft = '4px';
            }
            
            moreBtn.onclick = (e) => {
              e.preventDefault();
              e.stopPropagation();
              if (textWrapper.classList.contains('summary-abstract-collapsed')) {
                textWrapper.classList.remove('summary-abstract-collapsed');
                textWrapper.classList.add('summary-abstract-expanded');
                textWrapper.style.maxHeight = 'none';
                textWrapper.style.overflow = 'visible';
                moreBtn.textContent = 'Less';
                moreBtn.style.position = 'static';
                moreBtn.style.marginTop = '8px';
                moreBtn.style.marginLeft = '0';
                moreBtn.style.marginRight = '0';
                moreBtn.classList.remove('more-btn-overlay');
                
                // 展开时，如果有提示，添加到文本后面
                if (seeMoreHint && !textWrapper.contains(seeMoreHint)) {
                  textWrapper.appendChild(seeMoreHint);
                }
          } else {
                textWrapper.classList.remove('summary-abstract-expanded');
                textWrapper.classList.add('summary-abstract-collapsed');
                textWrapper.style.maxHeight = maxHeight + 'px';
                textWrapper.style.overflow = 'hidden';
                moreBtn.textContent = '...More';
                moreBtn.style.position = 'absolute';
                moreBtn.style.bottom = '12px';
                moreBtn.style.right = '12px';
                moreBtn.style.marginTop = '0';
                moreBtn.classList.add('more-btn-overlay');
                
                // 收起时，移除提示
                if (seeMoreHint && textWrapper.contains(seeMoreHint)) {
                  textWrapper.removeChild(seeMoreHint);
                }
              }
            };
          }
        };
        
        // 延迟检查，确保DOM已渲染
        setTimeout(checkAndAddButton, 50);
      } else {
        // AI生成或Wikipedia简单简介：检查是否需要翻译
        let summaryText = resource.summary;
        const lang = getCurrentLanguage();
        const dict = I18N_MAP[lang] || I18N_MAP['zh-CN'];
        
        // 检查是否是"这是关于{title}的百科文章。"格式
        const zhPattern = /^这是关于(.+)的百科文章。$/;
        const match = summaryText.match(zhPattern);
        if (match && lang === 'en-US') {
          // 提取标题并翻译
          const title = match[1];
          summaryText = dict['home.resources.wikipediaArticle'].replace('{title}', title);
        }
        
        summary.textContent = summaryText;
      }
      
      summaryContainer.appendChild(summary);
      contentArea.appendChild(summaryContainer);
    }
    
    item.appendChild(contentArea);
    
    // 按钮区域（右下角）
    if (resource.url) {
      const buttonArea = document.createElement('div');
      buttonArea.className = 'resource-item-actions';
      
      // 前往按钮
      const visitBtn = document.createElement('button');
      visitBtn.className = 'resource-action-btn resource-btn-visit';
      const visitText = getCurrentLanguage() === 'en-US' ? I18N_MAP['en-US']['home.resources.visit'] : I18N_MAP['zh-CN']['home.resources.visit'];
      visitBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg> ' + visitText;
      visitBtn.onclick = () => {
        window.open(resource.url, '_blank', 'noopener,noreferrer');
      };
      buttonArea.appendChild(visitBtn);
      
      // 下载按钮（视频不显示；Wikipedia 文本不显示；Google Scholar 仅显示前往）
      const isWikipedia = resource.source && resource.source.toLowerCase().includes('wikipedia');
      const isGoogleScholar = resource.source && resource.source.toLowerCase().includes('google scholar');
      if (type !== 'video' && !(type === 'txt' && isWikipedia) && !isGoogleScholar) {
        const downloadBtn = document.createElement('button');
        downloadBtn.className = 'resource-action-btn resource-btn-download';
        const downloadText = getCurrentLanguage() === 'en-US' ? I18N_MAP['en-US']['home.resources.download'] : I18N_MAP['zh-CN']['home.resources.download'];
        downloadBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg> ' + downloadText;
        downloadBtn.onclick = async () => {
          // 对于代码资源（GitHub仓库），直接下载zip文件
          if (type === 'code' && resource.url) {
            const githubUrl = resource.url;
            // 检查是否是GitHub URL
            const githubMatch = githubUrl.match(/github\.com\/([^\/]+)\/([^\/\?#]+)/);
            if (githubMatch) {
              const owner = githubMatch[1];
              const repo = githubMatch[2].replace(/\.git$/, '').split('/')[0]; // 移除.git后缀和可能的路径
              // GitHub zip下载URL：使用HEAD会自动下载默认分支（通常是main或master）
              const zipUrl = `https://github.com/${owner}/${repo}/archive/HEAD.zip`;
              
              // 创建一个隐藏的链接来触发下载
              const link = document.createElement('a');
              link.href = zipUrl;
              link.download = `${repo}.zip`;
              link.style.display = 'none';
              document.body.appendChild(link);
              link.click();
              document.body.removeChild(link);
              return;
            }
          }
          
          // 对于非GitHub代码资源或其他资源，使用原来的逻辑
          try {
            // 尝试通过fetch获取内容并下载
            const response = await fetch(resource.url);
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            // 根据资源类型设置文件名
            const extension = resource.url.match(/\.([^.]+)$/)?.[1] || 'html';
            const filename = `${(resource.title || 'resource').replace(/[<>:"/\\|?*]/g, '_')}.${extension}`;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            window.URL.revokeObjectURL(url);
    } catch (error) {
            // 如果fetch失败（跨域问题），直接打开链接
            console.warn('无法直接下载，打开链接:', error);
            window.open(resource.url, '_blank', 'noopener,noreferrer');
          }
        };
        buttonArea.appendChild(downloadBtn);
      }
      
      item.appendChild(buttonArea);
    }
    
    listElement.appendChild(item);
  });
}


// 工具函数
function showStatus(message, type = 'info') {
  uploadStatus.textContent = message;
  uploadStatus.className = `status-message ${type}`;
}

function updateProgress(percent, text, currentStep) {
  progressFill.style.width = percent + '%';
  const progressPercent = document.getElementById('progress-percent');
  if (progressPercent) {
    progressPercent.textContent = Math.round(percent) + '%';
  }
  
  // 翻译进度文本
  const lang = getCurrentLanguage();
  const dict = I18N_MAP[lang] || I18N_MAP['zh-CN'];
  let translatedText = text;
  
  if (text) {
    // 翻译"开始搜索相关资源..."
    if (text.includes('开始搜索相关资源')) {
      translatedText = lang === 'en-US' ? dict['home.processing.startSearching'] : text;
    }
    // 翻译"正在搜索文本、视频和代码资源..."
    else if (text.includes('正在搜索文本、视频和代码资源')) {
      translatedText = lang === 'en-US' ? dict['home.processing.searchingResources'] : text;
    }
    // 翻译"处理完成！"
    else if (text.includes('处理完成')) {
      translatedText = lang === 'en-US' ? dict['home.processing.complete'].replace('✨ ', '') : text;
    }
    // 翻译"准备开始..."
    else if (text.includes('准备开始')) {
      translatedText = lang === 'en-US' ? dict['home.processing.preparing'] : text;
    }
  }
  
  progressText.textContent = translatedText;
  
  // 更新步骤状态
  updateProgressSteps(percent, currentStep);
}

function updateProgressSteps(percent, currentStep) {
  const steps = document.querySelectorAll('.progress-step');
  
  // 根据步骤名称确定当前步骤索引
  const stepIndexMap = {
    'start': 0,
    'extract_keywords': 1,
    'keywords_extracted': 1,
    'search_resources': 2,
    'resources_found': 2,
    'save_results': 2,
    'results_saved': 2,
    'recommend': 3,
    'recommend_done': 3,
    'save_recommended': 3,
    'recommended_saved': 3,
    'prepare_data': 3,
    'complete': 4,
    'error': -1
  };
  
  let currentStepIndex = stepIndexMap[currentStep] !== undefined 
    ? stepIndexMap[currentStep] 
    : -1;
  
  // 如果无法从步骤名称确定，则根据进度百分比推断
  if (currentStepIndex === -1 && currentStep !== 'error') {
    if (percent < 25) {
      currentStepIndex = 0; // start
    } else if (percent < 50) {
      currentStepIndex = 1; // extract_keywords
    } else if (percent < 80) {
      currentStepIndex = 2; // search_resources
    } else if (percent < 100) {
      currentStepIndex = 3; // recommend
    } else {
      currentStepIndex = 4; // complete
    }
  }
  
  steps.forEach((step, index) => {
    const indicator = step.querySelector('.step-indicator');
    const status = step.querySelector('.step-status');
    
    if (currentStepIndex === -1) {
      // 错误状态
      step.classList.remove('active', 'completed', 'pending');
      indicator.classList.remove('active', 'completed');
      status.textContent = '错误';
    } else if (index < currentStepIndex) {
      // 已完成
      step.classList.add('completed');
      step.classList.remove('active', 'pending');
      indicator.classList.add('completed');
      indicator.classList.remove('active');
      const completedText = getCurrentLanguage() === 'en-US' ? I18N_MAP['en-US']['home.progress.status.completed'] : I18N_MAP['zh-CN']['home.progress.status.completed'];
      status.textContent = completedText;
    } else if (index === currentStepIndex) {
      // 进行中
      step.classList.add('active');
      step.classList.remove('completed', 'pending');
      indicator.classList.add('active');
      indicator.classList.remove('completed');
      const processingText = getCurrentLanguage() === 'en-US' ? I18N_MAP['en-US']['home.progress.status.processing'] : I18N_MAP['zh-CN']['home.progress.status.processing'];
      status.textContent = processingText;
    } else {
      // 等待中
      step.classList.add('pending');
      step.classList.remove('active', 'completed');
      indicator.classList.remove('active', 'completed');
      const pendingText = getCurrentLanguage() === 'en-US' ? I18N_MAP['en-US']['home.progress.status.pending'] : I18N_MAP['zh-CN']['home.progress.status.pending'];
      status.textContent = pendingText;
    }
  });
}

function updateProcessingDetails(text) {
  if (!text) return;
  
  const lang = getCurrentLanguage();
  const dict = I18N_MAP[lang] || I18N_MAP['zh-CN'];
  
  // 翻译常见的处理详情文本
  let translatedText = text;
  
  // 翻译"正在搜索文本、视频和代码资源..."
  if (text.includes('正在搜索文本、视频和代码资源')) {
    translatedText = lang === 'en-US' ? dict['home.processing.searchingResources'] : text;
  }
  // 翻译"开始搜索相关资源..."
  else if (text.includes('开始搜索相关资源')) {
    translatedText = lang === 'en-US' ? dict['home.processing.startSearching'] : text;
  }
  
  processingDetails.textContent = translatedText;
}

function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// 简单的中英文文案表
const I18N_MAP = {
  'zh-CN': {
    'nav.toc': '目录',
    'nav.home': '首页',
    'nav.help': '帮助',
    'nav.progress': '研发进度',
    'nav.aiEnhance': 'AI增强',
    'nav.contact': '加入我们',
    'home.title': 'AI多媒体推荐系统',
    'home.subtitle': '智能推荐，精确匹配',
    'home.step1.title': '上传文件',
    'home.step2.title': '处理中',
    'home.step3.title': '处理结果',
    'home.upload.hint': '上传zip文件',
    'home.upload.note': '支持.zip格式，需包含至少10个txt或pdf文档',
    'home.upload.button': '开始上传',
    'help.title': '帮助',
    'help.section.product': '关于产品',
    'help.section.tech': '关于技术',
    'help.section.innovation': '关于创新',
    'help.card.ourAim.title': '愿景',
    'help.card.ourAim.desc': '为AI学习者提供资源推荐服务',
    'help.card.quickStart.title': '快速开始',
    'help.card.quickStart.desc': '了解如何使用本系统',
    'help.card.features.title': '主要功能',
    'help.card.features.desc': '探索系统的核心功能',
    'help.card.workflow.title': '工作流程',
    'help.card.workflow.desc': '了解系统的工作流程和处理步骤',
    'help.card.nlp.title': '自然语言处理',
    'help.card.nlp.desc': '将繁杂输入转为可处理文本',
    'help.card.keyword.title': '关键词提取',
    'help.card.keyword.desc': '将文本信息提取精炼成关键词',
    'help.card.semantic.title': '语义相似度计算',
    'help.card.semantic.desc': '检查文档和资源是否匹配',
    'help.card.mmr.title': 'MMR',
    'help.card.mmr.desc': '去除部分意思相近的关键词',
    'help.card.recommend.title': '推荐算法',
    'help.card.recommend.desc': '推荐最合适的资源',
    'help.card.llm.title': 'OpenAI LLM',
    'help.card.llm.desc': '根据资源智能生成简介',
    'help.card.innovation.title': '产品优势',
    'help.card.innovation.desc': '相较同类竞品的优势',
    'help.card.techInnovation.title': '技术创新',
    'help.card.techInnovation.desc': '我们的技术创新点',
    'help.card.ux.title': '用户体验',
    'help.card.ux.desc': '更好的用户体验',
    'help.card.future.title': '未来规划',
    'help.card.future.desc': '系统的未来发展方向',
    'progress.title': '研发进度',
    'progress.subtitle': '持续创新，不断优化',
    'progress.overview.stage1': '项目启动',
    'progress.overview.stage2': '实现框架',
    'progress.overview.stage3': '方法与评估',
    'progress.overview.stage4': '优化与落地',
    'progress.overview.stage5': '交付',
    'progress.timeline.title': '研发时间线',
    'progress.timeline.stage1.date': '2025年10月 - 12月',
    'progress.timeline.stage1.title': '项目启动与需求分析',
    'progress.timeline.stage1.desc': '完成项目需求调研，确定技术栈和开发方向，制定详细的产品规划和技术架构设计。',
    'progress.timeline.stage1.tag1': '需求分析',
    'progress.timeline.stage1.tag2': '技术选型',
    'progress.timeline.stage1.tag3': '架构设计',
    'progress.timeline.stage2.date': '2026年1月 - 2月',
    'progress.timeline.stage2.title': '核心功能开发',
    'progress.timeline.stage2.desc': '实现关键词提取、资源搜索、推荐算法等核心功能模块开发，并对其进行基础测试与优化。',
    'progress.timeline.stage2.tag1': '关键词提取',
    'progress.timeline.stage2.tag2': '资源搜索',
    'progress.timeline.stage2.tag3': '推荐算法',
    'progress.timeline.stage2.tag4': '文件处理',
    'progress.timeline.stage3.date': '2026年3月',
    'progress.timeline.stage3.title': '用户界面设计与实现',
    'progress.timeline.stage3.desc': '设计并实现现代化的用户界面。完成网页开发，实现响应式布局和流畅的动画效果。',
    'progress.timeline.stage3.tag1': 'UI设计',
    'progress.timeline.stage3.tag2': '前端开发',
    'progress.timeline.stage3.tag3': '响应式布局',
    'progress.timeline.stage3.tag4': '动画效果',
    'progress.timeline.stage4.date': '2026年4月',
    'progress.timeline.stage4.title': '优化与落地',
    'progress.timeline.stage4.desc': '优化相关算法和系统性能，对网站进行整合和兼容性测试，并进一步提升用户体验。',
    'progress.timeline.stage4.tag1': '算法优化',
    'progress.timeline.stage4.tag2': '性能优化',
    'progress.timeline.stage4.tag3': '用户体验',
    'progress.timeline.stage4.tag4': '实时反馈',
    'progress.timeline.stage5.date': '2026年5月',
    'progress.timeline.stage5.title': '交付',
    'progress.timeline.stage5.desc': '交付网站，并进行后续的维护和优化。',
    'progress.timeline.stage6.date': '2026年6月',
    'progress.timeline.stage6.title': '未来规划',
    'progress.timeline.stage6.desc': '计划拓展移动端应用、API接口开放、社区功能等。持续收集用户反馈，不断迭代优化平台。',
    'progress.timeline.stage6.tag1': '移动端',
    'progress.timeline.stage6.tag2': 'API开放',
    'progress.timeline.stage6.tag3': '社区功能',
    'progress.timeline.stage6.tag4': '用户反馈',
    'progress.modules.title': '功能模块',
    'progress.modules.status.done': '已完成',
    'progress.modules.status.doing': '进行中',
    'progress.modules.status.todo': '计划中',
    'progress.modules.file.title': '文件处理',
    'progress.modules.file.desc': '支持ZIP文件上传、PDF自动转换、文本提取等功能，提供完整的文件处理流程。',
    'progress.modules.keyword.title': '关键词提取',
    'progress.modules.keyword.desc': '基于TF-IDF和TextRank算法，智能提取文档中的关键主题和概念。',
    'progress.modules.search.title': '多源搜索',
    'progress.modules.search.desc': '集成Wikipedia、YouTube、GitHub等多个数据源，提供全面的资源搜索能力。',
    'progress.modules.recommend.title': '智能推荐',
    'progress.modules.recommend.desc': '基于相似度计算的推荐算法，为用户推荐最相关的学习资源。',
    'progress.modules.ai.title': 'AI增强',
    'progress.modules.ai.desc': '集成大语言模型，提供智能摘要生成、内容理解等AI增强功能。',
    'progress.modules.lang.title': '多语言支持',
    'progress.modules.lang.desc': '支持中英文界面切换，未来将扩展更多语言支持。',
    'progress.modules.mobile.title': '移动端应用',
    'progress.modules.mobile.desc': '开发iOS和Android移动应用，让用户随时随地访问推荐系统。',
    'progress.modules.api.title': 'API接口',
    'progress.modules.api.desc': '提供RESTful API接口，支持第三方应用集成和二次开发。',
    'ai.title': 'AI增强',
    'ai.subtitle': '使用您的API密钥解锁更强大的AI功能',
    'ai.apikey.title': 'API密钥管理',
    'ai.apikey.desc': '您的API密钥仅存储在本地浏览器中，不会上传到服务器，确保您的隐私安全',
    'ai.apikey.status.none': '未设置',
    'ai.apikey.btn.save': '保存',
    'ai.apikey.btn.delete': '删除',
    'ai.feature.summary.title': '智能摘要',
    'ai.feature.summary.desc': '使用AI自动生成文档摘要，快速了解内容要点',
    'ai.feature.understand.title': '内容理解',
    'ai.feature.understand.desc': '深度理解文档内容，提供更精准的资源推荐',
    'ai.feature.qa.title': '智能问答',
    'ai.feature.qa.desc': '基于您的文档内容，提供智能问答服务',
    'ai.privacy.title': '隐私保护承诺',
    'ai.privacy.item1': '所有API密钥仅存储在您的浏览器本地，不会上传到服务器',
    'ai.privacy.item2': '我们不会收集、存储或传输您的API密钥信息',
    'ai.privacy.item3': '您可以随时删除已保存的API密钥',
    'ai.privacy.item4': '建议定期更换API密钥以确保安全',
    'contact.title': '加入我们',
    'contact.subtitle': '共创AI未来',
    'contact.contributors.title': '贡献者',
    'contact.form.name.label': '姓名 *',
    'contact.form.name.placeholder': '请输入您的姓名',
    'contact.form.email.label': '邮箱 *',
    'contact.form.subject.label': '主题 *',
    'contact.form.subject.placeholder': '请输入邮件主题',
    'contact.form.message.label': '动机 *',
    'contact.form.message.placeholder': '请输入您的动机...',
    'contact.form.submit': '发送',
    'contact.form.submitting': '发送中...',
    'home.progress.step.init': '初始化',
    'home.progress.step.extract': '提取关键词',
    'home.progress.step.search': '搜索资源',
    'home.progress.step.recommend': '推荐筛选',
    'home.progress.step.complete': '完成',
    'home.progress.status.pending': '等待中...',
    'home.progress.status.completed': '已完成',
    'home.progress.status.processing': '进行中...',
    'home.progress.status.error': '错误',
    'home.results.keywords': '提取的关键词/主题',
    'home.results.stats': '资源统计',
    'home.results.found': '找到的资源',
    'home.results.found.txt': '找到的文本资源',
    'home.results.found.video': '找到的视频资源',
    'home.results.found.code': '找到的代码资源',
    'home.results.recommended': '推荐的资源',
    'home.results.recommended.txt': '推荐的文本资源',
    'home.results.recommended.video': '推荐的视频资源',
    'home.results.recommended.code': '推荐的代码资源',
    'home.results.displayCount': '显示数量:',
    'home.panel.progress': '📊 处理进度',
    'home.panel.waiting': '等待开始处理...',
    'home.panel.resources': '📚 推荐资源',
    'home.resources.txt': '文本资源',
    'home.resources.video': '视频资源',
    'home.resources.code': '代码资源',
    'home.viewToggle.resourcesOnly': '仅显示推荐资源',
    'home.viewToggle.fullView': '显示完整视图',
    'home.upload.uploading': '正在上传...',
    'home.upload.success': '上传成功！',
    'home.upload.failed': '上传失败',
    'home.upload.invalidFormat': '请上传zip格式的文件',
    'home.upload.noFileSelected': '请先选择文件',
    'home.processing.start': '🚀 开始处理文件...',
    'home.processing.searching': '正在搜索关键词',
    'home.processing.complete': '✨ 处理完成！',
    'home.processing.allComplete': '所有资源已处理完成，推荐结果已生成。',
    'home.processing.preparing': '准备开始...',
    'home.resources.empty': '暂无推荐资源',
    'home.resources.source': '来源:',
    'home.resources.similarity': '相似度:',
    'home.resources.visit': '前往',
    'home.resources.download': '下载',
    'home.file.selected': '已选择文件:',
    'home.processing.searchingResources': '正在搜索文本、视频和代码资源...',
    'home.processing.startSearching': '开始搜索相关资源...',
    'home.resources.wikipediaArticle': '这是关于{title}的百科文章。'
  },
  'en-US': {
    'nav.toc': 'Menu',
    'nav.home': 'Home',
    'nav.help': 'Help',
    'nav.progress': 'Progress',
    'nav.aiEnhance': 'AI Enhance',
    'nav.contact': 'Join Us',
    'home.title': 'Reinforcement Ebook System',
    'home.subtitle': 'Intelligent recommendation, precise matching',
    'home.step1.title': 'Upload Files',
    'home.step2.title': 'Processing',
    'home.step3.title': 'Results',
    'home.upload.hint': 'Upload a zip file',
    'home.upload.note': 'Supports .zip containing at least 10 txt or pdf documents',
    'home.upload.button': 'Start Upload',
    'help.title': 'Help',
    'help.section.product': 'About Product',
    'help.section.tech': 'About Technology',
    'help.section.innovation': 'About Innovation',
    'help.card.ourAim.title': 'Vision',
    'help.card.ourAim.desc': 'Recommend AI resources for learners',
    'help.card.quickStart.title': 'Quick Start',
    'help.card.quickStart.desc': 'Learn how to use this system for your learning',
    'help.card.features.title': 'Key Features',
    'help.card.features.desc': 'Explore the core capabilities of the system',
    'help.card.workflow.title': 'Workflow',
    'help.card.workflow.desc': 'Understand the processing pipeline of the system',
    'help.card.nlp.title': 'NLP',
    'help.card.nlp.desc': 'Transform complex inputs into manageable text',
    'help.card.keyword.title': 'Keyword Extraction',
    'help.card.keyword.desc': 'Extract keywords from text information',
    'help.card.semantic.title': 'Semantic Similarity',
    'help.card.semantic.desc': 'Check if the document and resource match',
    'help.card.mmr.title': 'MMR',
    'help.card.mmr.desc': 'Remove some similar keywords',
    'help.card.recommend.title': 'Recommender',
    'help.card.recommend.desc': 'Recommend the most suitable resource',
    'help.card.llm.title': 'OpenAI LLM',
    'help.card.llm.desc': 'Generate concise descriptions for resources',
    'help.card.innovation.title': 'Unique Advantage',
    'help.card.innovation.desc': 'Our unique advantages over competitors',
    'help.card.techInnovation.title': 'Tech Innovation',
    'help.card.techInnovation.desc': 'Our main Tech Innovations',
    'help.card.ux.title': 'User Experience',
    'help.card.ux.desc': 'Better UX design',
    'help.card.future.title': 'Future Plan',
    'help.card.future.desc': 'Future directions of the system',
    'progress.title': 'Our Progress',
    'progress.subtitle': 'Keep innovating and optimizing',
    'progress.overview.stage1': 'Kick-off',
    'progress.overview.stage2': 'Framework design',
    'progress.overview.stage3': 'Methods & Evaluation',
    'progress.overview.stage4': 'Optimization & Deployment',
    'progress.overview.stage5': 'Delivery',
    'progress.timeline.title': 'Development Timeline',
    'progress.timeline.stage1.date': 'Oct–Dec 2025',
    'progress.timeline.stage1.title': 'Project Kick-off & Requirements',
    'progress.timeline.stage1.desc': 'Complete requirement research, choose the tech stack, and design detailed product and architecture.',
    'progress.timeline.stage1.tag1': 'Requirements',
    'progress.timeline.stage1.tag2': 'Tech Stack',
    'progress.timeline.stage1.tag3': 'Architecture',
    'progress.timeline.stage2.date': 'Jan–Feb 2026',
    'progress.timeline.stage2.title': 'Core Feature Development',
    'progress.timeline.stage2.desc': 'Implement keyword extraction, resource search and recommendation modules, plus initial testing and optimisation.',
    'progress.timeline.stage2.tag1': 'Keyword Extraction',
    'progress.timeline.stage2.tag2': 'Resource Search',
    'progress.timeline.stage2.tag3': 'Recommendation',
    'progress.timeline.stage2.tag4': 'File Processing',
    'progress.timeline.stage3.date': 'Mar 2026',
    'progress.timeline.stage3.title': 'UI Design & Implementation',
    'progress.timeline.stage3.desc': 'Design and implement a modern UI with responsive layout and smooth animations.',
    'progress.timeline.stage3.tag1': 'UI Design',
    'progress.timeline.stage3.tag2': 'Frontend Dev',
    'progress.timeline.stage3.tag3': 'Responsive Layout',
    'progress.timeline.stage3.tag4': 'Animation',
    'progress.timeline.stage4.date': 'Apr 2026',
    'progress.timeline.stage4.title': 'Optimization & Landing',
    'progress.timeline.stage4.desc': 'Optimise algorithms and performance, integrate modules, and enhance user experience.',
    'progress.timeline.stage4.tag1': 'Algorithm Optimisation',
    'progress.timeline.stage4.tag2': 'Performance',
    'progress.timeline.stage4.tag3': 'UX',
    'progress.timeline.stage4.tag4': 'Real-time Feedback',
    'progress.timeline.stage5.date': 'May 2026',
    'progress.timeline.stage5.title': 'Delivery',
    'progress.timeline.stage5.desc': 'Deliver the website and continue maintenance and optimisation.',
    'progress.timeline.stage6.date': 'Jun 2026',
    'progress.timeline.stage6.title': 'Future Plan',
    'progress.timeline.stage6.desc': 'Plan mobile apps, open APIs and community features, and iterate based on user feedback.',
    'progress.timeline.stage6.tag1': 'Mobile',
    'progress.timeline.stage6.tag2': 'Open API',
    'progress.timeline.stage6.tag3': 'Community',
    'progress.timeline.stage6.tag4': 'User Feedback',
    'progress.modules.title': 'Functional Modules',
    'progress.modules.status.done': 'Completed',
    'progress.modules.status.doing': 'In Progress',
    'progress.modules.status.todo': 'Planned',
    'progress.modules.file.title': 'File Processing',
    'progress.modules.file.desc': 'Support ZIP upload, PDF conversion and text extraction to form a complete processing pipeline.',
    'progress.modules.keyword.title': 'Keyword Extraction',
    'progress.modules.keyword.desc': 'Use TF-IDF and TextRank to extract key topics and concepts.',
    'progress.modules.search.title': 'Multi-source Search',
    'progress.modules.search.desc': 'Integrate Wikipedia, YouTube, GitHub and other sources for comprehensive search.',
    'progress.modules.recommend.title': 'Intelligent Recommendation',
    'progress.modules.recommend.desc': 'Recommend the most relevant learning resources based on similarity scoring.',
    'progress.modules.ai.title': 'AI Enhance',
    'progress.modules.ai.desc': 'Integrate LLM to provide smart summarisation and content understanding.',
    'progress.modules.lang.title': 'Multi-language Support',
    'progress.modules.lang.desc': 'Support Chinese/English UI and more languages in future.',
    'progress.modules.mobile.title': 'Mobile Apps',
    'progress.modules.mobile.desc': 'Plan iOS and Android apps so users can access the system anywhere.',
    'progress.modules.api.title': 'API Interfaces',
    'progress.modules.api.desc': 'Provide RESTful APIs for third-party integration and secondary development.',
    'ai.title': 'AI Enhance',
    'ai.subtitle': 'Use your API key to unlock advanced AI features',
    'ai.apikey.title': 'API Key Management',
    'ai.apikey.desc': 'Your API key is stored only in this browser and never sent to our server.',
    'ai.apikey.status.none': 'Not set',
    'ai.apikey.btn.save': 'Save',
    'ai.apikey.btn.delete': 'Delete',
    'ai.feature.summary.title': 'Smart Summary',
    'ai.feature.summary.desc': 'Automatically generate concise summaries so you can grasp the gist quickly.',
    'ai.feature.understand.title': 'Content Understanding',
    'ai.feature.understand.desc': 'Deeply understand document content to provide more accurate recommendations.',
    'ai.feature.qa.title': 'Smart Q&A',
    'ai.feature.qa.desc': 'Answer questions based on the content of your documents.',
    'ai.privacy.title': 'Privacy Commitment',
    'ai.privacy.item1': 'All API keys are safely stored locally.',
    'ai.privacy.item2': 'We do not collect, store or transmit your API keys.',
    'ai.privacy.item3': 'You can delete saved API keys at any time.',
    'ai.privacy.item4': 'We recommend rotating API keys regularly for better security.',
    'contact.title': 'Join Us',
    'contact.subtitle': 'Shape the future of AI',
    'contact.contributors.title': 'Contributors',
    'contact.form.name.label': 'Name *',
    'contact.form.name.placeholder': 'Please enter your name',
    'contact.form.email.label': 'Email *',
    'contact.form.subject.label': 'Subject *',
    'contact.form.subject.placeholder': 'Please enter the subject',
    'contact.form.message.label': 'Motivation *',
    'contact.form.message.placeholder': 'Please describe your motivation',
    'contact.form.submit': 'Send',
    'contact.form.submitting': 'Sending...',
    'home.progress.step.init': 'Initialization',
    'home.progress.step.extract': 'Extract Keywords',
    'home.progress.step.search': 'Search Resources',
    'home.progress.step.recommend': 'Recommendation Filtering',
    'home.progress.step.complete': 'Complete',
    'home.progress.status.pending': 'Pending...',
    'home.progress.status.completed': 'Completed',
    'home.progress.status.processing': 'In Progress...',
    'home.progress.status.error': 'Error',
    'home.results.keywords': 'Extracted Keywords/Topics',
    'home.results.stats': 'Resource Statistics',
    'home.results.found': 'Found Resources',
    'home.results.found.txt': 'Found Text Resources',
    'home.results.found.video': 'Found Video Resources',
    'home.results.found.code': 'Found Code Resources',
    'home.results.recommended': 'Recommended Resources',
    'home.results.recommended.txt': 'Recommended Text Resources',
    'home.results.recommended.video': 'Recommended Video Resources',
    'home.results.recommended.code': 'Recommended Code Resources',
    'home.results.displayCount': 'Display Count:',
    'home.panel.progress': '📊 Processing Progress',
    'home.panel.waiting': 'Waiting to start processing...',
    'home.panel.resources': '📚 Recommended Resources',
    'home.resources.txt': 'Text Resources',
    'home.resources.video': 'Video Resources',
    'home.resources.code': 'Code Resources',
    'home.viewToggle.resourcesOnly': 'Resources Only',
    'home.viewToggle.fullView': 'Full View',
    'home.upload.uploading': 'Uploading...',
    'home.upload.success': 'Upload successful!',
    'home.upload.failed': 'Upload failed',
    'home.upload.invalidFormat': 'Please upload a zip file',
    'home.upload.noFileSelected': 'Please select a file first',
    'home.processing.start': '🚀 Start processing files...',
    'home.processing.searching': 'Searching for keywords',
    'home.processing.complete': '✨ Processing complete!',
    'home.processing.allComplete': 'All resources have been processed and recommendations have been generated.',
    'home.processing.preparing': 'Preparing to start...',
    'home.resources.empty': 'No recommended resources',
    'home.resources.source': 'Source:',
    'home.resources.similarity': 'Similarity:',
    'home.resources.visit': 'Visit',
    'home.resources.download': 'Download',
    'home.file.selected': 'File selected:',
    'home.processing.searchingResources': 'Searching text, video, and code resources...',
    'home.processing.startSearching': 'Start searching for relevant resources...',
    'home.resources.wikipediaArticle': 'This is a Wikipedia article about {title}.'
  }
};

function getCurrentLanguage() {
  const htmlRoot = document.getElementById('html-root');
  if (htmlRoot) {
    const lang = htmlRoot.getAttribute('lang');
    if (lang) return lang;
  }
  const saved = localStorage.getItem('language');
  return saved || 'zh-CN';
}

function applyLanguage(lang) {
  const htmlRoot = document.getElementById('html-root');
  if (htmlRoot) {
    htmlRoot.setAttribute('lang', lang);
  }
  const languageText = document.getElementById('language-text');
  if (languageText) {
    languageText.textContent = lang === 'zh-CN' ? '中文' : 'English';
  }

  const dict = I18N_MAP[lang];
  if (!dict) return;

  const elements = document.querySelectorAll('[data-i18n-key]');
  elements.forEach(el => {
    const key = el.getAttribute('data-i18n-key');
    const text = dict[key];
    if (text) {
      // 支持纯文本 / HTML / 属性三种模式
      const attr = el.getAttribute('data-i18n-attr');
      if (attr) {
        el.setAttribute(attr, text);
      } else if (el.hasAttribute('data-i18n-html')) {
        el.innerHTML = text;
      } else {
        el.textContent = text;
      }
    }
  });
  
  // 更新动态内容（进度步骤状态、视图切换按钮等）
  updateDynamicContent(lang);
}

function updateDynamicContent(lang) {
  const dict = I18N_MAP[lang];
  if (!dict) return;
  
  // 更新进度步骤状态
  const progressSteps = document.querySelectorAll('.progress-step');
  progressSteps.forEach(step => {
    const status = step.querySelector('.step-status');
    if (status) {
      if (step.classList.contains('completed')) {
        status.textContent = dict['home.progress.status.completed'] || status.textContent;
      } else if (step.classList.contains('active')) {
        status.textContent = dict['home.progress.status.processing'] || status.textContent;
      } else {
        status.textContent = dict['home.progress.status.pending'] || status.textContent;
      }
    }
  });
  
  // 更新视图切换按钮文本
  const viewToggleText = document.getElementById('view-toggle-text');
  if (viewToggleText) {
    const isResourcesOnly = document.getElementById('main-container')?.classList.contains('resources-only');
    if (isResourcesOnly) {
      viewToggleText.textContent = dict['home.viewToggle.fullView'] || viewToggleText.textContent;
    } else {
      viewToggleText.textContent = dict['home.viewToggle.resourcesOnly'] || viewToggleText.textContent;
    }
  }
  
  // 更新终端输出中的等待文本
  const terminalOutput = document.getElementById('terminal-output');
  if (terminalOutput && terminalOutput.children.length === 1) {
    const firstLine = terminalOutput.querySelector('.terminal-line');
    if (firstLine && firstLine.getAttribute('data-i18n-key') === 'home.panel.waiting') {
      firstLine.textContent = dict['home.panel.waiting'] || firstLine.textContent;
    }
  }
  
  // 更新进度文本
  const progressText = document.getElementById('progress-text');
  if (progressText && progressText.textContent) {
    const currentText = progressText.textContent;
    let translatedText = currentText;
    
    // 翻译常见的进度文本
    if (currentText.includes('开始搜索相关资源')) {
      translatedText = dict['home.processing.startSearching'] || currentText;
    } else if (currentText.includes('正在搜索文本、视频和代码资源')) {
      translatedText = dict['home.processing.searchingResources'] || currentText;
    } else if (currentText.includes('处理完成')) {
      translatedText = dict['home.processing.complete'].replace('✨ ', '') || currentText;
    } else if (currentText.includes('准备开始')) {
      translatedText = dict['home.processing.preparing'] || currentText;
    }
    
    progressText.textContent = translatedText;
  }
  
  // 更新处理详情文本
  const processingDetails = document.getElementById('processing-details');
  if (processingDetails && processingDetails.textContent) {
    updateProcessingDetails(processingDetails.textContent);
  }
  
  // 更新资源列表中的文本（来源、相似度、按钮等）
  const resourceItems = document.querySelectorAll('.resource-item');
  resourceItems.forEach(item => {
    // 更新来源标签
    const sourceSpan = item.querySelector('.resource-item-source span');
    if (sourceSpan) {
      sourceSpan.textContent = dict['home.resources.source'] || sourceSpan.textContent;
    }
    
    // 更新相似度标签
    const similaritySpan = item.querySelector('.resource-item-similarity');
    if (similaritySpan) {
      const similarityValue = similaritySpan.textContent.match(/[\d.]+%/);
      if (similarityValue) {
        similaritySpan.textContent = `${dict['home.resources.similarity']} ${similarityValue[0]}`;
      }
    }
    
    // 更新按钮文本
    const visitBtn = item.querySelector('.resource-btn-visit');
    if (visitBtn) {
      const svg = visitBtn.querySelector('svg');
      if (svg) {
        visitBtn.innerHTML = svg.outerHTML + ' ' + (dict['home.resources.visit'] || '前往');
      }
    }
    
    const downloadBtn = item.querySelector('.resource-btn-download');
    if (downloadBtn) {
      const svg = downloadBtn.querySelector('svg');
      if (svg) {
        downloadBtn.innerHTML = svg.outerHTML + ' ' + (dict['home.resources.download'] || '下载');
      }
    }
    
    // 更新摘要中的"这是关于...的百科文章"文本
    const summary = item.querySelector('.resource-item-summary');
    if (summary) {
      const summaryText = summary.textContent;
      const zhPattern = /^这是关于(.+)的百科文章。$/;
      const match = summaryText.match(zhPattern);
      if (match && lang === 'en-US') {
        const title = match[1];
        summary.textContent = dict['home.resources.wikipediaArticle'].replace('{title}', title);
      } else if (lang === 'zh-CN' && summaryText.includes('This is a Wikipedia article about')) {
        // 如果从英文切回中文，也需要处理
        const enPattern = /^This is a Wikipedia article about (.+)\.$/;
        const enMatch = summaryText.match(enPattern);
        if (enMatch) {
          const title = enMatch[1];
          summary.textContent = dict['home.resources.wikipediaArticle'].replace('{title}', title);
        }
      }
    }
  });
}

// 导航栏功能
function setupNavigation() {
  // 语言切换
  const languageBtn = document.getElementById('language-btn');
  const languageText = document.getElementById('language-text');
  const htmlRoot = document.getElementById('html-root');
  
  if (languageBtn && languageText && htmlRoot) {
    languageBtn.addEventListener('click', () => {
      const currentLang = htmlRoot.getAttribute('lang') || 'zh-CN';
      const newLang = currentLang === 'zh-CN' ? 'en-US' : 'zh-CN';
      localStorage.setItem('language', newLang);
      applyLanguage(newLang);
    });
  }
  
  // 主题切换
  const themeBtn = document.getElementById('theme-btn');
  const themeIconMoon = document.getElementById('theme-icon-moon');
  const themeIconSun = document.getElementById('theme-icon-sun');
  
  if (themeBtn && themeIconMoon && themeIconSun) {
    const updateThemeIcon = (isLightMode) => {
      if (isLightMode) {
        themeIconMoon.style.display = 'none';
        themeIconSun.style.display = 'block';
      } else {
        themeIconMoon.style.display = 'block';
        themeIconSun.style.display = 'none';
      }
    };
    
    themeBtn.addEventListener('click', () => {
      const body = document.body;
      const isLightMode = body.classList.contains('light-mode');
      
      // 直接切换主题，让背景色自然渐变
      if (isLightMode) {
        body.classList.remove('light-mode');
        localStorage.setItem('theme', 'dark');
        updateThemeIcon(false);
      } else {
        body.classList.add('light-mode');
        localStorage.setItem('theme', 'light');
        updateThemeIcon(true);
      }
    });
    
    // 初始化主题图标 - 基于保存的主题设置
    const savedTheme = localStorage.getItem('theme');
    updateThemeIcon(savedTheme === 'light');
  }
  
  // 导航链接点击（当前页面标记为active）
  const navLinks = document.querySelectorAll('.nav-link');
  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      // 如果链接指向当前页面，阻止默认行为
      if (link.getAttribute('data-page') === 'home' && window.location.pathname === '/') {
        e.preventDefault();
      }
      
      // 更新active状态
      navLinks.forEach(l => l.classList.remove('active'));
      link.classList.add('active');
    });
  });

  // 移动端目录按钮下拉
  const menuToggle = document.getElementById('nav-menu-btn');
  if (menuToggle) {
    menuToggle.addEventListener('click', () => {
      document.body.classList.toggle('nav-menu-open');
    });

    // 点击导航链接后自动收起菜单（仅移动端）
    navLinks.forEach(link => {
      link.addEventListener('click', () => {
        if (window.innerWidth <= 768) {
          document.body.classList.remove('nav-menu-open');
        }
      });
    });

    // 窗口尺寸变化时清理状态
    window.addEventListener('resize', () => {
      if (window.innerWidth > 768) {
        document.body.classList.remove('nav-menu-open');
      }
    });
  }
}

// 加载用户偏好设置
function loadPreferences() {
  // 加载语言设置
  const savedLanguage = localStorage.getItem('language');
  const lang = savedLanguage || 'zh-CN';
  applyLanguage(lang);
  
  // 加载主题设置
  const savedTheme = localStorage.getItem('theme');
  const themeIconMoon = document.getElementById('theme-icon-moon');
  const themeIconSun = document.getElementById('theme-icon-sun');
  
  if (savedTheme === 'light') {
    document.body.classList.add('light-mode');
    if (themeIconMoon && themeIconSun) {
      themeIconMoon.style.display = 'none';
      themeIconSun.style.display = 'block';
    }
  } else {
    document.body.classList.remove('light-mode');
    if (themeIconMoon && themeIconSun) {
      themeIconMoon.style.display = 'block';
      themeIconSun.style.display = 'none';
    }
  }
}

// 设置视图切换功能
function setupViewToggle() {
  if (!viewToggleBtn || !viewToggleFloat) return;
  
  // 使用事件委托，因为按钮内容可能会被替换
  viewToggleFloat.addEventListener('click', (e) => {
    if (e.target.closest('.view-toggle-btn')) {
      toggleResourcesOnlyView();
    }
  });
  
  // 加载保存的视图偏好（延迟执行，确保resultsPanel已显示）
  setTimeout(() => {
    const savedView = localStorage.getItem('resourcesOnlyView');
    if (savedView === 'true' && mainContainer.classList.contains('has-results') && resultsPanel.style.display !== 'none') {
      toggleResourcesOnlyView();
    }
  }, 100);
}

// 切换仅显示推荐资源视图
function toggleResourcesOnlyView() {
  if (!mainContainer || !viewToggleBtn) return;
  
  isResourcesOnlyView = !isResourcesOnlyView;
  
  if (isResourcesOnlyView) {
    // 切换到仅显示推荐资源模式
    mainContainer.classList.add('resources-only');
    
    // 更新按钮图标和文本
    const fullViewText = getCurrentLanguage() === 'en-US' ? I18N_MAP['en-US']['home.viewToggle.fullView'] : I18N_MAP['zh-CN']['home.viewToggle.fullView'];
    viewToggleBtn.innerHTML = `
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
        <line x1="9" y1="3" x2="9" y2="21"></line>
      </svg>
      <span id="view-toggle-text">${fullViewText}</span>
    `;
  } else {
    // 切换回完整视图
    mainContainer.classList.remove('resources-only');
    
    // 更新按钮图标和文本
    const resourcesOnlyText = getCurrentLanguage() === 'en-US' ? I18N_MAP['en-US']['home.viewToggle.resourcesOnly'] : I18N_MAP['zh-CN']['home.viewToggle.resourcesOnly'];
    viewToggleBtn.innerHTML = `
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="3" y="3" width="7" height="7"></rect>
        <rect x="14" y="3" width="7" height="7"></rect>
        <rect x="14" y="14" width="7" height="7"></rect>
        <rect x="3" y="14" width="7" height="7"></rect>
      </svg>
      <span id="view-toggle-text">${resourcesOnlyText}</span>
    `;
  }
  
  // 保存视图偏好
  localStorage.setItem('resourcesOnlyView', isResourcesOnlyView.toString());
  
  // 重新获取viewToggleText引用
  viewToggleText = document.getElementById('view-toggle-text');
}
