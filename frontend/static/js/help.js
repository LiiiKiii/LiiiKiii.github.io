// 帮助页面JavaScript

// 详细内容数据
const helpDetails = {
  'our-aim': {
    title: '愿景',
    content: `
      <p>我们致力于创建智能化的AI学习资源推荐系统，帮助学习者快速找到最合适的学习资源。</p>
      <p>系统依据上传资料从多个来源收集资料，以最大程度帮助用户学习相关知识。</p>
      <h3>核心目标</h3>
      <ul>
        <li>节省时间：自动筛选和推荐，无需手动搜索</li>
        <li>全面覆盖：整合文本、视频、代码等多种资源类型</li>
        <li>智能匹配：基于语义相似度，确保推荐质量</li>
        <li>持续更新：实时获取最新的学术和开源资源</li>
      </ul>
    `
  },
  'how-to-use': {
    title: '快速开始',
    content: `
      <p>按下面四步即可完成一次完整体验。</p>
      <ol>
        <li><strong>准备zip</strong>：把与同一学习主题相关的txt / pdf打包成一个zip文件</li>
        <li><strong>上传文件</strong>：在首页点击“上传zip文件”，选择刚才的zip</li>
        <li><strong>等待处理</strong>：系统自动提取关键词并搜索外部资源</li>
        <li><strong>浏览推荐</strong>：查看推荐的文本、视频和代码链接，按需要打开或收藏</li>
      </ol>
    `
  },
  'features': {
    title: '主要功能',
    content: `
      <p>本系统提供了丰富的功能，帮助您高效地使用系统并获取学习资源。</p>
      <h3>主要功能</h3>
      <ul>
        <li><strong>智能关键词提取</strong>：使用TF-IDF和语义分析技术，自动识别文档中的关键主题</li>
        <li><strong>多源资源搜索</strong>：从Wikipedia、ArXiv、YouTube、GitHub等多个平台搜索资源</li>
        <li><strong>智能推荐算法</strong>：基于相似度评分，推荐最相关的资源</li>
        <li><strong>实时处理进度</strong>：实时显示处理进度和当前搜索的关键词</li>
        <li><strong>资源分类展示</strong>：按文本、视频、代码分类展示推荐结果</li>
        <li><strong>动态数量调整</strong>：通过滑块动态调整每种资源的显示数量</li>
        <li><strong>AI智能摘要</strong>：使用OpenAI API生成资源简介，帮助快速了解内容</li>
        <li><strong>资源下载</strong>：支持直接下载文本资源和GitHub代码仓库</li>
      </ul>
    `
  },
  'workflow': {
    title: '工作流程',
    content: `
      <p>系统的工作流程经过精心设计，确保推荐结果的准确性和相关性。</p>
      <h3>处理流程</h3>
      <ol>
        <li><strong>文档上传与解析</strong>：接收zip文件，解压并提取所有txt和pdf文档，将pdf转换为文本</li>
        <li><strong>关键词提取</strong>：分析所有文档内容，使用TF-IDF和语义评分提取最重要的关键词</li>
        <li><strong>资源搜索</strong>：对每个关键词，在多个平台上搜索相关资源</li>
        <li><strong>资源推荐</strong>：计算每个资源与文档的相似度，选择最相关的资源进行推荐</li>
        <li><strong>摘要生成</strong>：为推荐的资源生成智能摘要，帮助用户快速了解内容</li>
        <li><strong>结果展示</strong>：将推荐结果分类展示，支持用户交互和资源访问</li>
      </ol>
    `
  },
  'ai-technology': {
    title: 'AI Technology',
    content: `
      <p>本系统采用了多种先进的AI技术，确保推荐结果的准确性和相关性。</p>
      <h3>核心技术</h3>
      <ul>
        <li><strong>自然语言处理（NLP）</strong>：文本预处理、分词、去停用词等</li>
        <li><strong>TF-IDF向量化</strong>：将文档转换为向量表示，用于相似度计算</li>
        <li><strong>语义相似度计算</strong>：使用余弦相似度衡量文档和资源之间的相关性</li>
        <li><strong>最大边际相关性（MMR）</strong>：确保推荐的关键词具有多样性和代表性</li>
        <li><strong>OpenAI GPT模型</strong>：生成智能的资源摘要和描述</li>
      </ul>
      <h3>技术优势</h3>
      <ul>
        <li>结合统计方法和语义理解，提高推荐准确性</li>
        <li>支持多语言处理，适应不同语言的文档</li>
        <li>实时处理能力，快速响应用户需求</li>
      </ul>
    `
  },
  'keyword-extraction': {
    title: 'Keyword Extraction',
    content: `
      <p>关键词提取是系统的核心功能之一，直接影响推荐结果的质量。</p>
      <h3>提取方法</h3>
      <ul>
        <li><strong>TF-IDF评分</strong>：衡量词在文档中的重要性，过滤常见词汇</li>
        <li><strong>语义评分</strong>：基于AI领域词典，提升专业术语的权重</li>
        <li><strong>组合评分</strong>：综合TF-IDF和语义评分，选择最具代表性的关键词</li>
        <li><strong>去噪处理</strong>：过滤引用格式、地址、版权信息等无关内容</li>
      </ul>
      <h3>优化策略</h3>
      <ul>
        <li>限制关键词长度，避免过长的短语</li>
        <li>去除重复和相似的关键词</li>
        <li>优先选择在多个文档中出现的词汇</li>
        <li>考虑关键词的语义相关性</li>
      </ul>
    `
  },
  'recommendation': {
    title: '推荐算法',
    content: `
      <p>推荐算法确保用户获得最相关和最有价值的资源。</p>
      <h3>推荐原理</h3>
      <ul>
        <li><strong>相似度计算</strong>：使用余弦相似度比较资源内容与用户文档的相似性</li>
        <li><strong>多维度评分</strong>：综合考虑标题、内容、描述等多个维度</li>
        <li><strong>类型平衡</strong>：确保文本、视频、代码资源的均衡推荐</li>
        <li><strong>质量筛选</strong>：过滤低质量和无关的资源</li>
      </ul>
      <h3>推荐策略</h3>
      <ul>
        <li>优先推荐相似度高的资源</li>
        <li>考虑资源的来源和权威性</li>
        <li>平衡新旧资源，既有经典内容也有最新进展</li>
        <li>支持用户自定义显示数量</li>
      </ul>
    `
  },
  'openai-llm': {
    title: 'OpenAI LLM',
    content: `
      <p>OpenAI 大语言模型只做一件事：把复杂内容变成短而好懂的简介。</p>
      <ul>
        <li><strong>摘要</strong>：为每条推荐资源生成几句话的说明</li>
        <li><strong>补充判断</strong>：在相似度接近时，帮助区分更合适的资源</li>
        <li><strong>可选功能</strong>：你有 OpenAI API Key 时可以开启，没有也不影响系统使用</li>
      </ul>
    `
  },
  'nlp': {
    title: '自然语言处理',
    content: `
      <p>自然语言处理技术在系统中发挥着关键作用。</p>
      <h3>NLP应用</h3>
      <ul>
        <li><strong>文本预处理</strong>：清理HTML标签、特殊字符、格式化文本</li>
        <li><strong>内容提取</strong>：从网页和文档中提取纯文本内容</li>
        <li><strong>摘要生成</strong>：使用AI模型生成简洁的资源描述</li>
        <li><strong>语义理解</strong>：理解文档和资源的语义内容，提高匹配准确性</li>
      </ul>
      <h3>处理流程</h3>
      <ol>
        <li>原始文本获取</li>
        <li>文本清理和标准化</li>
        <li>关键词和主题提取</li>
        <li>语义分析和匹配</li>
        <li>处理结果用于后续算法</li>
      </ol>
    `
  },
  'semantic-similarity': {
    title: '语义相似度计算',
    content: `
      <p>语义相似度计算是系统推荐算法的核心，用于衡量文档和资源之间的相关性。</p>
      <h3>计算方法</h3>
      <ul>
        <li><strong>余弦相似度</strong>：通过计算向量之间的夹角来衡量相似性，范围在0到1之间</li>
        <li><strong>TF-IDF向量化</strong>：将文档和资源转换为向量表示，便于相似度计算</li>
        <li><strong>多维度比较</strong>：综合考虑标题、内容、描述等多个维度的相似性</li>
        <li><strong>阈值筛选</strong>：设置相似度阈值（≥0.05），过滤低相关性资源</li>
      </ul>
      <h3>应用场景</h3>
      <ul>
        <li>文档与资源的匹配度评估</li>
        <li>推荐结果的排序和筛选</li>
        <li>关键词与资源的相关性判断</li>
        <li>多源资源的统一评分</li>
      </ul>
    `
  },
  'mmr': {
    title: 'MMR（最大边际相关性）',
    content: `
      <p>MMR算法用于确保推荐的关键词具有多样性和代表性，避免关键词过于相似。</p>
      <h3>算法原理</h3>
      <ul>
        <li><strong>多样性平衡</strong>：在选择关键词时，既要考虑重要性，也要考虑与已选关键词的差异性</li>
        <li><strong>边际相关性</strong>：衡量新关键词相对于已选关键词集合的独特价值</li>
        <li><strong>优化目标</strong>：在相关性和多样性之间找到最佳平衡点</li>
      </ul>
      <h3>算法优势</h3>
      <ul>
        <li>避免关键词重复和冗余</li>
        <li>提高关键词集合的覆盖范围</li>
        <li>确保推荐结果的多样性</li>
        <li>提升搜索资源的全面性</li>
      </ul>
    `
  },
  'innovation-concept': {
    title: '产品优势',
    content: `
      <p>我们不是通用聊天机器人，而是为 AI 学习场景定制的“资源推荐助手”。</p>
      <ul>
        <li><strong>对比 NotebookLM</strong>：不做问答，只显示整合式推荐列表，并提供更多更强推荐</li>
        <li><strong>对比 ChatGPT</strong>：更关注“给你哪些具体资源”而不是长篇回答</li>
        <li><strong>专注一个领域</strong>：只做 AI / ML 学习资源，让推荐更聚焦</li>
      </ul>
    `
  },
  'technical-innovation': {
    title: '技术创新',
    content: `
      <p>系统在技术层面实现了多项创新，提升了推荐质量和用户体验。</p>
      <h3>技术创新点</h3>
      <ul>
        <li><strong>多源整合</strong>：首次整合Wikipedia、ArXiv、YouTube、GitHub等多个平台资源</li>
        <li><strong>智能推荐</strong>：结合TF-IDF，语义评分与反向比对以达到最佳性能</li>
        <li><strong>实时处理</strong>：采用SSE技术实现实时进度显示，提升用户体验</li>
        <li><strong>LLM摘要生成</strong>：可选使用OpenAI API为资源生成摘要</li>
      </ul>
      <h3>技术优势</h3>
      <ul>
        <li>高效的文档处理能力</li>
        <li>准确的相似度匹配算法</li>
        <li>流畅的用户交互体验</li>
        <li>可扩展的架构设计</li>
      </ul>
    `
  },
  'user-experience': {
    title: '用户体验',
    content: `
      <p>系统在用户体验设计上进行了多方面的创新和优化。</p>
      <h3>体验优化</h3>
      <ul>
        <li><strong>直观的界面</strong>：简洁清晰的设计，让用户一目了然</li>
        <li><strong>实时反馈</strong>：处理过程中实时显示进度和当前状态</li>
        <li><strong>灵活控制</strong>：支持动态调整资源显示数量</li>
        <li><strong>便捷操作</strong>：一键下载、快速访问资源链接</li>
      </ul>
      <h3>交互设计</h3>
      <ul>
        <li>拖拽上传，操作简单</li>
        <li>分类展示，信息清晰</li>
        <li>响应式设计，适配多种设备</li>
        <li>流畅的动画效果，提升视觉体验</li>
      </ul>
    `
  },
  'future-development': {
    title: '未来规划',
    content: `
      <p>系统将持续发展和改进，为用户提供更好的服务。</p>
      <h3>发展方向</h3>
      <ul>
        <li><strong>功能扩展</strong>：增加更多资源类型和来源平台</li>
        <li><strong>算法优化</strong>：持续改进推荐算法，提高推荐准确性</li>
        <li><strong>性能提升</strong>：优化处理速度，支持更大规模的文档处理</li>
        <li><strong>用户体验</strong>：根据用户反馈不断优化界面和交互</li>
      </ul>
      <h3>计划功能</h3>
      <ul>
        <li>用户账户系统，保存历史记录</li>
        <li>个性化推荐偏好设置</li>
        <li>多语言支持</li>
        <li>移动端应用开发</li>
        <li>API接口开放</li>
      </ul>
    `
  }
};

// 英文版详细内容（只在英文模式下使用）
const helpDetailsEn = {
  'our-aim': {
    title: 'Vision',
    content: `
      <p>We aim to build an intelligent recommendation system that helps AI learners quickly find the most suitable learning resources.</p>
      <p>The system aggregates content from multiple sources based on your uploaded documents to maximise learning efficiency.</p>
      <h3>Core goals</h3>
      <ul>
        <li>Save time: automatic filtering and recommendation without manual search</li>
        <li>Broad coverage: integrate text, video and code resources</li>
        <li>Smart matching: semantic similarity to ensure recommendation quality</li>
        <li>Continuous updates: keep up with the latest academic and open‑source resources</li>
      </ul>
    `
  },
  'how-to-use': {
    title: 'Quick Start',
    content: `
      <p>Follow the four simple steps below for a complete experience.</p>
      <ol>
        <li><strong>Prepare a zip</strong>: pack txt / pdf files on the same topic into one zip</li>
        <li><strong>Upload</strong>: on the home page click “Upload a zip file” and choose the zip</li>
        <li><strong>Wait for processing</strong>: the system extracts keywords and searches external resources</li>
        <li><strong>Browse recommendations</strong>: review text, video and code links, then open or bookmark what you need</li>
      </ol>
    `
  },
  'features': {
    title: 'Key Features',
    content: `
      <p>The system provides a rich set of features to help you efficiently obtain learning resources.</p>
      <h3>Main features</h3>
      <ul>
        <li><strong>Smart keyword extraction</strong>: automatically identify key topics using TF‑IDF and semantic analysis</li>
        <li><strong>Multi‑source search</strong>: query platforms such as Wikipedia, ArXiv, YouTube and GitHub</li>
        <li><strong>Recommendation algorithm</strong>: rank and recommend the most relevant resources</li>
        <li><strong>Real‑time progress</strong>: show processing status and current keywords</li>
        <li><strong>Categorised display</strong>: group recommendations by text, video and code</li>
        <li><strong>Dynamic count control</strong>: adjust how many items are shown per type</li>
        <li><strong>AI summaries</strong>: generate concise descriptions for faster understanding</li>
        <li><strong>Resource download</strong>: download text and GitHub repositories directly</li>
      </ul>
    `
  },
  'workflow': {
    title: 'Workflow',
    content: `
      <p>The workflow is carefully designed to ensure accurate and relevant recommendations.</p>
      <h3>Processing pipeline</h3>
      <ol>
        <li><strong>Upload & parsing</strong>: receive the zip, extract txt and pdf files, convert pdf to text</li>
        <li><strong>Keyword extraction</strong>: analyse all documents and extract important keywords</li>
        <li><strong>Resource search</strong>: search multiple platforms for each keyword</li>
        <li><strong>Recommendation</strong>: compute similarity between documents and resources and select the best ones</li>
        <li><strong>Summary generation</strong>: generate AI summaries for recommended resources</li>
        <li><strong>Result display</strong>: show results by category with interactive controls</li>
      </ol>
    `
  },
  'ai-technology': {
    title: 'AI Technology',
    content: `
      <p>This system uses multiple advanced AI techniques to ensure accurate and relevant recommendations.</p>
      <h3>Core techniques</h3>
      <ul>
        <li><strong>NLP</strong>: text preprocessing, tokenisation and stop‑word filtering</li>
        <li><strong>TF‑IDF vectorisation</strong>: turn documents into vectors for similarity computation</li>
        <li><strong>Semantic similarity</strong>: cosine similarity between documents and resources</li>
        <li><strong>MMR</strong>: keep selected keywords diverse and representative</li>
        <li><strong>OpenAI GPT</strong>: generate intelligent summaries and descriptions</li>
      </ul>
    `
  },
  'keyword-extraction': {
    title: 'Keyword Extraction',
    content: `
      <p>Keyword extraction is one of the core functions and directly affects recommendation quality.</p>
      <h3>Methods</h3>
      <ul>
        <li><strong>TF‑IDF scoring</strong>: measure word importance and filter common words</li>
        <li><strong>Semantic scoring</strong>: up‑weight AI / ML domain terms</li>
        <li><strong>Combined scoring</strong>: select the most representative keywords</li>
        <li><strong>Denoising</strong>: remove references, addresses and boilerplate</li>
      </ul>
    `
  },
  'semantic-similarity': {
    title: 'Semantic Similarity',
    content: `
      <p>Semantic similarity is at the core of our recommendation algorithm.</p>
      <h3>How we compute it</h3>
      <ul>
        <li><strong>Cosine similarity</strong>: compare vector angles in the 0–1 range</li>
        <li><strong>TF‑IDF vectorisation</strong>: turn documents and resources into vectors</li>
        <li><strong>Multi‑dimensional comparison</strong>: consider title, content and description</li>
        <li><strong>Threshold filtering</strong>: drop resources below a similarity threshold</li>
      </ul>
    `
  },
  'mmr': {
    title: 'MMR (Maximum Marginal Relevance)',
    content: `
      <p>MMR is used to ensure diversity and representativeness in the selected keywords.</p>
      <h3>Idea</h3>
      <ul>
        <li><strong>Diversity vs relevance</strong>: balance importance and difference to existing keywords</li>
        <li><strong>Marginal gain</strong>: measure the unique value of a new keyword</li>
      </ul>
    `
  },
  'recommendation': {
    title: 'Recommendation Algorithm',
    content: `
      <p>The recommendation algorithm ensures you get the most relevant and valuable resources.</p>
      <h3>Principles</h3>
      <ul>
        <li><strong>Similarity</strong>: cosine similarity between resources and your documents</li>
        <li><strong>Multi‑dimensional scoring</strong>: combine title, content and description signals</li>
        <li><strong>Type balancing</strong>: keep a healthy mix of text, video and code</li>
      </ul>
    `
  },
  'openai-llm': {
    title: 'OpenAI LLM',
    content: `
      <p>The OpenAI large language model turns complex content into short, easy‑to‑understand summaries.</p>
      <ul>
        <li><strong>Summaries</strong>: generate a few sentences for each recommended resource</li>
        <li><strong>Extra judgement</strong>: help choose the better resource when similarity scores are close</li>
        <li><strong>Optional feature</strong>: can be enabled with your own OpenAI API key; the system still works without it</li>
      </ul>
    `
  },
  'nlp': {
    title: 'Natural Language Processing',
    content: `
      <p>Natural language processing plays a key role in the system.</p>
      <h3>NLP applications</h3>
      <ul>
        <li><strong>Text preprocessing</strong>: clean HTML tags, special characters and format text</li>
        <li><strong>Content extraction</strong>: extract plain text from web pages and documents</li>
        <li><strong>Summary generation</strong>: use AI models to generate concise resource descriptions</li>
        <li><strong>Semantic understanding</strong>: understand the semantic content of documents and resources to improve matching accuracy</li>
      </ul>
      <h3>Processing pipeline</h3>
      <ol>
        <li>Raw text acquisition</li>
        <li>Text cleaning and normalisation</li>
        <li>Keyword and topic extraction</li>
        <li>Semantic analysis and matching</li>
        <li>Results used in subsequent algorithms</li>
      </ol>
    `
  },
  'innovation-concept': {
    title: 'Unique Advantage',
    content: `
      <p>Instead of being a general‑purpose chatbot, this system is a “resource recommendation assistant” tailored for AI learning.</p>
      <ul>
        <li><strong>Compared with NotebookLM</strong>: No general chat, only structured & clear recommendations</li>
        <li><strong>Compared with ChatGPT</strong>: focus on accurate recommendations rather than long answers</li>
        <li><strong>Focused domain</strong>: specialise in AI / ML resources to keep recommendations targeted</li>
      </ul>
    `
  },
  'technical-innovation': {
    title: 'Tech Innovation',
    content: `
      <p>The system introduces several Tech Innovations to improve recommendation quality and user experience.</p>
      <h3>Innovation points</h3>
      <ul>
        <li><strong>Multimedia integration</strong>: combine resources from Wikipedia, ArXiv, YouTube, GitHub and more</li>
        <li><strong>Smart recommendation</strong>: combine TF‑IDF, semantic scoring with reverse comparison to achieve best performance</li>
        <li><strong>Real‑time progress</strong>: SSE‑based progress streaming</li>
        <li><strong>AI summarisation</strong>: use OpenAI to generate concise summaries</li>
      </ul>
    `
  },
  'user-experience': {
    title: 'User Experience',
    content: `
      <p>We have made many UX improvements and innovations.</p>
      <h3>Experience optimisation</h3>
      <ul>
        <li><strong>Intuitive interface</strong>: clean layout so users understand at a glance</li>
        <li><strong>Real‑time feedback</strong>: show progress and status during processing</li>
        <li><strong>Flexible control</strong>: dynamically adjust how many resources are shown</li>
        <li><strong>Convenient operations</strong>: one‑click download and quick access</li>
      </ul>
      <h3>Interaction design</h3>
      <ul>
        <li>Drag‑and‑drop upload for simplicity</li>
        <li>Categorised display for clarity</li>
        <li>Responsive design for different devices</li>
        <li>Fluid animations for better visual experience</li>
      </ul>
    `
  },
  'future-development': {
    title: 'Future Plan',
    content: `
      <p>The system will continue to evolve to provide better service.</p>
      <h3>Directions</h3>
      <ul>
        <li><strong>More features</strong>: add new resource types and sources</li>
        <li><strong>Algorithm improvements</strong>: continuously refine recommendation quality</li>
        <li><strong>Performance</strong>: support larger document sets and faster processing</li>
        <li><strong>User experience</strong>: iterate UI and interactions based on feedback</li>
      </ul>
      <h3>Planned features</h3>
      <ul>
        <li>User accounts and history</li>
        <li>Personalised recommendation preferences</li>
        <li>Broader multi‑language support</li>
      </ul>
    `
  }
};

function getCurrentLanguage() {
  // 以页面上的 lang 属性为主，保证和当前界面一致
  const htmlRoot = document.getElementById('html-root');
  const attr = htmlRoot && htmlRoot.getAttribute('lang');
  if (attr) return attr;
  const saved = localStorage.getItem('language');
  return saved || 'zh-CN';
}

// 初始化
document.addEventListener('DOMContentLoaded', () => {
  // 首先加载偏好设置，确保主题在页面渲染前就应用
  loadPreferences();
  // 触发页面进入动画
  triggerPageAnimation();
  initCarousels();
  initModal();
  setupNavigation();
  // 初始化滚动渐入效果
  initScrollFadeIn();
});

// 触发页面进入动画
function triggerPageAnimation() {
  // 确保容器可见
  const container = document.querySelector('.help-container');
  if (container) {
    container.style.opacity = '1';
  }
  
  // 强制触发动画 - 确保所有元素都能正确显示
  setTimeout(() => {
    const hero = document.querySelector('.help-hero');
    const productTitle = document.querySelector('#product-carousel')?.closest('.help-section')?.querySelector('.help-section-title');
    const aiTitle = document.querySelector('#ai-carousel')?.closest('.help-section')?.querySelector('.help-section-title');
    const productCards = document.querySelectorAll('#product-carousel .help-card');
    const aiCards = document.querySelectorAll('#ai-carousel .help-card');
    const buttons = document.querySelectorAll('.help-carousel-btn');
    
    // 如果动画没有触发，强制显示（fallback）
    setTimeout(() => {
      if (hero && parseFloat(getComputedStyle(hero).opacity) < 0.1) {
        hero.style.opacity = '1';
        hero.style.transform = 'translateY(0)';
      }
      if (productTitle && parseFloat(getComputedStyle(productTitle).opacity) < 0.1) {
        productTitle.style.opacity = '1';
        productTitle.style.transform = 'translateY(0)';
      }
      productCards.forEach(card => {
        if (parseFloat(getComputedStyle(card).opacity) < 0.1) {
          card.style.opacity = '1';
          card.style.transform = 'translateY(0) translateX(0)';
        }
      });
      if (aiTitle && parseFloat(getComputedStyle(aiTitle).opacity) < 0.1) {
        aiTitle.style.opacity = '1';
        aiTitle.style.transform = 'translateY(0)';
      }
      aiCards.forEach(card => {
        if (parseFloat(getComputedStyle(card).opacity) < 0.1) {
          card.style.opacity = '1';
          card.style.transform = 'translateY(0) translateX(0)';
        }
      });
      buttons.forEach(btn => {
        if (parseFloat(getComputedStyle(btn).opacity) < 0.1) {
          btn.style.opacity = '1';
        }
      });
    }, 2000); // 2秒后检查，如果还没显示就强制显示
  }, 100);
}

// 初始化轮播
function initCarousels() {
  const carousels = document.querySelectorAll('.help-carousel');
  
  carousels.forEach(carousel => {
    const wrapper = carousel.closest('.help-carousel-wrapper');
    const prevBtn = wrapper.querySelector('.help-carousel-btn-prev');
    const nextBtn = wrapper.querySelector('.help-carousel-btn-next');
    
    // 更新按钮状态
    const updateButtons = () => {
      const isAtStart = carousel.scrollLeft <= 0;
      const isAtEnd = carousel.scrollLeft >= carousel.scrollWidth - carousel.clientWidth - 10;
      
      prevBtn.disabled = isAtStart;
      nextBtn.disabled = isAtEnd;
    };
    
    // 初始状态
    updateButtons();
    
    // 滚动时更新按钮
    carousel.addEventListener('scroll', updateButtons);
    
    // 窗口大小改变时更新
    window.addEventListener('resize', updateButtons);
    
    // 按钮点击事件
    prevBtn.addEventListener('click', (e) => {
      e.preventDefault();
      // 取消任何正在进行的平滑滚动
      carousel.style.scrollBehavior = 'auto';
      carousel.scrollBy({ left: -340, behavior: 'auto' });
      setTimeout(() => {
        carousel.style.scrollBehavior = '';
      }, 0);
    });
    
    nextBtn.addEventListener('click', (e) => {
      e.preventDefault();
      // 取消任何正在进行的平滑滚动
      carousel.style.scrollBehavior = 'auto';
      carousel.scrollBy({ left: 340, behavior: 'auto' });
      setTimeout(() => {
        carousel.style.scrollBehavior = '';
      }, 0);
    });
    
    // 鼠标悬停时自动滚动
    let scrollInterval = null;
    const scrollSpeed = 8; // 每次滚动的像素数（减小步长）
    const scrollIntervalTime = 10; // 滚动间隔时间（毫秒）
    
    // 停止滚动的函数
    const stopScrolling = () => {
      if (scrollInterval) {
        clearInterval(scrollInterval);
        scrollInterval = null;
      }
      // 确保取消任何平滑滚动
      carousel.style.scrollBehavior = 'auto';
    };
    
    prevBtn.addEventListener('mouseenter', () => {
      if (prevBtn.disabled) return;
      stopScrolling(); // 先停止之前的滚动
      // 取消平滑滚动行为
      carousel.style.scrollBehavior = 'auto';
      scrollInterval = setInterval(() => {
        if (carousel.scrollLeft > 0) {
          carousel.scrollLeft -= scrollSpeed;
          updateButtons(); // 更新按钮状态
        } else {
          stopScrolling();
        }
      }, scrollIntervalTime);
    });
    
    prevBtn.addEventListener('mouseleave', stopScrolling);
    
    nextBtn.addEventListener('mouseenter', () => {
      if (nextBtn.disabled) return;
      stopScrolling(); // 先停止之前的滚动
      // 取消平滑滚动行为
      carousel.style.scrollBehavior = 'auto';
      scrollInterval = setInterval(() => {
        const maxScroll = carousel.scrollWidth - carousel.clientWidth;
        if (carousel.scrollLeft < maxScroll) {
          carousel.scrollLeft += scrollSpeed;
          updateButtons(); // 更新按钮状态
        } else {
          stopScrolling();
        }
      }, scrollIntervalTime);
    });
    
    nextBtn.addEventListener('mouseleave', stopScrolling);
    
    // 触摸滑动支持
    let isDown = false;
    let startX;
    let scrollLeft;
    
    carousel.addEventListener('mousedown', (e) => {
      isDown = true;
      carousel.style.cursor = 'grabbing';
      startX = e.pageX - carousel.offsetLeft;
      scrollLeft = carousel.scrollLeft;
    });
    
    carousel.addEventListener('mouseleave', () => {
      isDown = false;
      carousel.style.cursor = 'grab';
    });
    
    carousel.addEventListener('mouseup', () => {
      isDown = false;
      carousel.style.cursor = 'grab';
    });
    
    carousel.addEventListener('mousemove', (e) => {
      if (!isDown) return;
      e.preventDefault();
      const x = e.pageX - carousel.offsetLeft;
      const walk = (x - startX) * 2;
      carousel.scrollLeft = scrollLeft - walk;
    });
    
    carousel.style.cursor = 'grab';
  });
}

// 初始化模态框
function initModal() {
  const modal = document.getElementById('help-modal');
  const closeBtn = modal.querySelector('.help-modal-close');
  const overlay = modal.querySelector('.help-modal-overlay');
  const cards = document.querySelectorAll('.help-card');
  
  // 卡片点击事件
  cards.forEach(card => {
    card.addEventListener('click', () => {
      const detailId = card.getAttribute('data-detail');
      if (detailId && helpDetails[detailId]) {
        showDetail(detailId);
      }
    });
  });
  
  // 关闭按钮
  closeBtn.addEventListener('click', closeModal);
  overlay.addEventListener('click', closeModal);
  
  // ESC键关闭
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.classList.contains('active')) {
      closeModal();
    }
  });
}

// 显示详细内容
function showDetail(detailId) {
  const lang = getCurrentLanguage();
  let detail = helpDetails[detailId];
  if (lang === 'en-US' && helpDetailsEn[detailId]) {
    detail = helpDetailsEn[detailId];
  }
  if (!detail) return;
  
  const modal = document.getElementById('help-modal');
  const modalBody = document.getElementById('help-modal-body');
  
  modalBody.innerHTML = `
    <h2>${detail.title}</h2>
    ${detail.content}
  `;
  
  modal.classList.add('active');
  document.body.style.overflow = 'hidden';
}

// 关闭模态框
function closeModal() {
  const modal = document.getElementById('help-modal');
  modal.classList.remove('active');
  document.body.style.overflow = '';
}

// 初始化滚动渐入效果
function initScrollFadeIn() {
  // 处理AI部分
  setupSectionFadeIn('#ai-carousel');
  // 处理创新部分
  setupSectionFadeIn('#innovation-carousel');
}

// 通用的section渐入效果设置
function setupSectionFadeIn(carouselId) {
  const section = document.querySelector(carouselId)?.closest('.help-section');
  if (!section) return;
  
  const title = section.querySelector('.help-section-title');
  const cards = section.querySelectorAll(`${carouselId} .help-card`);
  const buttons = section.querySelectorAll('.help-carousel-btn');
  
  // 初始状态：完全透明
  title.style.opacity = '0';
  title.style.transform = 'translateY(50px)';
  cards.forEach(card => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(50px) scale(0.95)';
  });
  buttons.forEach(btn => {
    btn.style.opacity = '0';
    btn.style.transform = 'translateY(20px)';
  });
  
  const handleScroll = () => {
    const rect = section.getBoundingClientRect();
    const windowHeight = window.innerHeight;
    
    // 计算元素进入视口的进度
    // 当元素顶部进入视口80%位置时开始显示
    const elementTop = rect.top;
    const elementHeight = rect.height;
    const triggerPoint = windowHeight * 0.8;
    const fadeInRange = windowHeight * 0.6; // 渐入范围

    // 如果页面已经滚动到接近底部，强制该 section 完全显示
    const doc = document.documentElement;
    const scrollBottom = window.scrollY + window.innerHeight;
    const pageHeight = doc.scrollHeight;
    const nearPageBottom = (pageHeight - scrollBottom) < 40;

    if (nearPageBottom && rect.top < windowHeight) {
      title.style.opacity = '1';
      title.style.transform = 'translateY(0)';
      cards.forEach(card => {
        card.style.opacity = '1';
        card.style.transform = 'translateY(0) scale(1)';
      });
      buttons.forEach(btn => {
        btn.style.opacity = '1';
        btn.style.transform = 'translateY(0)';
      });
      return;
    }
    
    if (elementTop < triggerPoint && elementTop > -elementHeight) {
      // 计算渐入进度 (0 到 1)
      // 当元素顶部从triggerPoint移动到triggerPoint - fadeInRange时，progress从0到1
      const progress = Math.max(0, Math.min(1, (triggerPoint - elementTop) / fadeInRange));
      
      // 标题渐入（稍快一些）
      const titleProgress = Math.max(0, Math.min(1, progress * 1.1));
      title.style.opacity = titleProgress.toString();
      title.style.transform = `translateY(${50 * (1 - titleProgress)}px)`;
      
      // 卡片同时从下往上渐入（移除从左到右的延迟）
      const cardProgress = Math.max(0, Math.min(1, (progress - 0.15) * 1.2)); // 标题显示后开始
      cards.forEach(card => {
        card.style.opacity = cardProgress.toString();
        const translateY = 50 * (1 - cardProgress);
        const scale = 0.95 + (0.05 * cardProgress);
        card.style.transform = `translateY(${translateY}px) scale(${scale})`;
      });
      
      // 按钮渐入（在卡片显示后）
      const buttonProgress = Math.max(0, Math.min(1, (progress - 0.4) * 2));
      buttons.forEach(btn => {
        btn.style.opacity = buttonProgress.toString();
        btn.style.transform = `translateY(${20 * (1 - buttonProgress)}px)`;
      });
    } else if (elementTop <= -elementHeight) {
      // 元素完全滚过，确保完全显示
      title.style.opacity = '1';
      title.style.transform = 'translateY(0)';
      cards.forEach(card => {
        card.style.opacity = '1';
        card.style.transform = 'translateY(0) scale(1)';
      });
      buttons.forEach(btn => {
        btn.style.opacity = '1';
        btn.style.transform = 'translateY(0)';
      });
    } else if (elementTop > triggerPoint) {
      // 元素还未进入触发区域，保持隐藏
      title.style.opacity = '0';
      title.style.transform = 'translateY(50px)';
      cards.forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(50px) scale(0.95)';
      });
      buttons.forEach(btn => {
        btn.style.opacity = '0';
        btn.style.transform = 'translateY(20px)';
      });
    }
  };
  
  // 使用 requestAnimationFrame 优化滚动性能
  let ticking = false;
  const scrollHandler = () => {
    if (!ticking) {
      window.requestAnimationFrame(() => {
        handleScroll();
        ticking = false;
      });
      ticking = true;
    }
  };
  
  window.addEventListener('scroll', scrollHandler, { passive: true });
  
  // 初始检查
  handleScroll();
}
