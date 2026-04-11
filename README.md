# AI-Pedia

<div align="center">

**AI-Powered Multimedia Resource Recommendation System**

An intelligent resource recommendation system focused on AI/Machine Learning domain, capable of automatically extracting keywords from user-uploaded documents, searching multi-source external resources, and filtering the most relevant learning resources using content-based recommendation algorithms.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-Apache--2.0-yellow.svg)](LICENSE)

[Features](#features) • [Quick Start](#quick-start) • [Project Structure](#project-structure) • [Tech Stack](#tech-stack) • [Usage](#usage)

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Core Algorithms](#core-algorithms)
- [Development Guide](#development-guide)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

**AI-Pedia** is an intelligent AI/Machine Learning resource recommendation system. The system can:

- **Document Analysis**: Extract key themes and keywords from user-uploaded AI/ML related documents
- **Multi-Source Search**: Automatically search resources from multiple platforms including Wikipedia, Google Scholar, arXiv, YouTube, GitHub, etc.
- **Intelligent Recommendation**: Use Content-Based Filtering (CBF) algorithm to filter resources most relevant to user documents
- **AI Enhancement**: Optionally use OpenAI API to generate resource summaries for smarter content understanding
- **Content Cleaning**: Automatically filter non-AI related content, contact information, and other irrelevant information

## Features

### Core Features

- **AI Domain Specialized**: Focused on AI/Machine Learning domain, intelligently filtering non-AI related content
- **Multi-Format Support**: Supports TXT and PDF documents (PDF automatically converted to TXT)
- **Intelligent Keyword Extraction**: Uses TF-IDF and MMR algorithms to extract key themes
- **Multi-Source Resource Search**:
  - **Text Resources**: Wikipedia, Google Scholar, arXiv, etc.
  - **Video Resources**: YouTube educational videos
  - **Code Resources**: GitHub, Google Colab, Kaggle, Stack Overflow, etc.
- **CBF Recommendation System**: Filters most relevant resources based on content similarity
- **AI Summary Generation**: Optionally use OpenAI API to generate intelligent summaries
- **Smart Fallback**: Automatically uses rule-based summary generation when API fails
- **Content Cleaning**: Automatically removes contact information, department info, and other irrelevant content
- **Result Packaging**: Automatically packages recommendation results as ZIP files

### User Interface

- **Modern UI**: Apple-style design with light/dark theme switching
- **Multi-Language Support**: Supports Chinese/English interface switching
- **Responsive Design**: Adapts to different screen sizes
- **Real-Time Progress**: Real-time display of processing progress and status

## Tech Stack

### Backend

- **Python 3.8+**
- **Flask 2.0+** - Web framework
- **scikit-learn** - Machine learning algorithms (TF-IDF, similarity calculation)
- **numpy** - Numerical computation
- **requests** - HTTP requests
- **pdfplumber / PyPDF2** - PDF processing
- **openai** - AI summary generation (optional)

### Frontend

- **HTML5 / CSS3** - Page structure and styling
- **JavaScript (ES6+)** - Interactive logic
- **Server-Sent Events (SSE)** - Real-time progress updates

### Core Algorithms

- **TF-IDF** - Keyword extraction and document vectorization
- **MMR (Maximal Marginal Relevance)** - Keyword diversity selection
- **Cosine Similarity** - Content similarity calculation
- **CBF (Content-Based Filtering)** - Content-based recommendation

## Quick Start

### Requirements

- Python 3.8 or higher
- pip package manager
- Stable network connection (for accessing external resources)

### Installation Steps

1. **Clone the repository**

```bash
git clone https://github.com/LiiiKiii/Reinforcement-Ebook-System.git
cd Reinforcement-Ebook-System
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Start the application**

**Method 1: Using startup script (Recommended)**

```bash
chmod +x start.sh
./start.sh
```

**Method 2: Manual startup**

```bash
python app.py
```

4. **Access the application**

Open your browser and visit: `http://localhost:5000`

---

## Online Demo (GitHub Pages)

This project also provides a **pure frontend static version** that can be hosted on GitHub Pages to showcase the UI without running the Flask backend locally.

- When deploying Pages from the `main` branch **root folder**:
  - The root `index.html` simply redirects `https://<username>.github.io/` to `https://<username>.github.io/frontend/`
  - The actual static site entry is `frontend/index.html` under the repository
- The static site will:
  - Fully render the modern UI, theme switch, i18n, animations, etc.
  - But any backend‑dependent features (ZIP upload, keyword extraction, recommendation pipeline, contact form submission, etc.) will **not** actually run on GitHub Pages unless you point the JS API calls to a separately deployed Flask backend

## Configuration

### OpenAI API Key (Optional)

The system supports using OpenAI API to generate intelligent summaries for resources. After configuring the API Key, the system will generate more accurate and informative summaries.

#### Configuration Method

1. **Get API Key**

Visit [OpenAI Platform](https://platform.openai.com/api-keys) to register and obtain an API Key

2. **Set environment variable**

**Linux/Mac:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Windows:**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

3. **Or set in startup script**

```bash
export OPENAI_API_KEY="your-api-key-here"
python app.py
```

> **Note**: If no API Key is configured, the system will automatically use intelligent fallback methods to generate summaries, and all features remain fully functional.

### Port Configuration

If the default port 5000 is occupied, you can modify the last line in `app.py`:

```python
app.run(debug=True, port=5001)  # Change to another port
```

## Usage

### 1. Prepare Documents

Prepare a folder containing at least 10 TXT or PDF documents, preferably on AI/Machine Learning topics.

**Document Requirements:**
- Format: TXT or PDF
- Quantity: At least 10 documents
- Topic: Recommended to be AI/Machine Learning related (e.g., deep learning, neural networks, natural language processing, etc.)
- Packaging: Compress the folder into ZIP format

### 2. Upload and Process

1. Open your browser and visit `http://localhost:5000`
2. Click the upload area and select your ZIP file
3. The system will automatically verify the file count (at least 10 documents)
4. After successful upload, the system will automatically start processing:
   - Extract keywords
   - Search external resources
   - Calculate similarity
   - Generate recommendation results

### 3. Download Results

After processing is complete, click the "Download Recommendation Results" button, and the system will download a ZIP file containing:

- `txt/` - Recommended text resources (irrelevant content already cleaned)
- `video/` - Recommended video links and descriptions
- `code/` - Recommended code resources (GitHub, Colab, etc.)

## Project Structure

```
AI-Pedia/Project/
├── app.py                      # Flask main application entry
├── requirements.txt            # Python dependencies list
├── README.md / README_CH.md    # Project documentation (EN / CN)
├── start.sh                    # Startup script
├── LICENSE                     # License file
├── index.html                  # Root redirect page (redirects to /frontend/, for GitHub Pages)
│
├── frontend/                   # Frontend code (shared by Flask + GitHub Pages)
│   ├── index.html              # Pure static home page (GitHub Pages entry)
│   ├── templates/              # HTML templates for Flask rendering
│   │   ├── index.html          # Main page
│   │   ├── help.html           # Help page
│   │   ├── progress.html       # Progress page
│   │   ├── ai-enhance.html     # AI Enhancement page
│   │   └── contact.html        # Contact page
│   └── static/                 # Static resources
│       ├── css/                # Styles
│       │   ├── style.css       # Main styles
│       │   ├── help.css        # Help page styles
│       │   ├── progress.css    # Progress page styles
│       │   ├── ai-enhance.css  # AI Enhancement page styles
│       │   └── contact.css     # Contact page styles
│       ├── js/                 # JavaScript files
│       │   ├── main.js         # Global logic (upload, SSE, i18n, theme, etc.)
│       │   ├── help.js         # Help page logic
│       │   ├── progress.js     # Progress page logic
│       │   ├── ai-enhance.js   # AI Enhancement page logic
│       │   └── contact.js      # Contact page logic
│       └── images/
│           └── home-icon.png   # Site icon
│
├── backend/                    # Backend code
│   ├── core/                   # Core functional modules
│   │   ├── __init__.py
│   │   ├── keyword_extractor.py    # Keyword extraction module
│   │   ├── resource_searcher.py    # Resource search module
│   │   ├── recommender.py          # CBF recommendation system
│   │   └── ai_summarizer.py        # AI summary generation module
│   └── utils/                  # Utility modules
│       ├── __init__.py
│       └── file_utils.py       # File processing utilities
│
├── data/                       # Data directory (auto-created)
│   ├── uploads/                # User uploaded files
│   ├── results/                # Search results
│   └── outputs/                # Final outputs
│
├── test/                       # Testing & evaluation
│   ├── README.md               # Test overview
│   ├── performance/            # Performance tests
│   │   └── run_performance_tests.py
│   ├── evaluation_subjective/  # Subjective evaluation (questionnaires, score templates)
│   └── evaluation_objective/   # Objective metrics (performance + quality metrics & tables)
│
└── docs/                       # Documentation directory
    ├── STRUCTURE.md            # Project structure documentation
    ├── TECHNICAL_OVERVIEW.md   # Technical overview
    ├── EVALUATION.md           # Evaluation documentation
    ├── INNOVATION.md           # Innovation points
    └── REFERENCES.md           # References and resources
```

---

## Testing & Evaluation

The project includes a dedicated `test/` directory to support both **qualitative** and **quantitative** evaluation:

- **`test/README.md`**: High‑level overview of the testing framework and directory layout  
- **`test/performance/run_performance_tests.py`**:
  - Unified entry for basic performance testing of core modules such as `keyword_extractor` and `recommender`
  - Designed to be extended with additional tests for `resource_searcher` and `ai_summarizer`
- **`test/evaluation_subjective/`**:
  - Templates and guidance for subjective evaluation (usability, satisfaction, trust, etc.)
  - Suitable for Likert‑scale questionnaires filled by users/experts
- **`test/evaluation_objective/`**:
  - Definitions and formulas for objective metrics (coverage, redundancy, noise rate, HitRate@K, etc.)
  - Ready‑to‑use table templates for recording experiment results and plotting figures

## Core Algorithms

### Keyword Extraction

1. **TF-IDF Vectorization**: Convert documents to TF-IDF vectors
2. **MMR Algorithm**: Use Maximal Marginal Relevance to ensure keyword diversity and representativeness
3. **Noise Filtering**: Automatically filter URLs, emails, contact information, institutional information, etc.
4. **Default Extraction**: Extract 10 AI-related keywords

### Resource Recommendation

1. **Document Vectorization**: Convert user documents and external resources to TF-IDF vectors
2. **Similarity Calculation**: Use cosine similarity to calculate content similarity
3. **Threshold Filtering**: Only recommend resources with similarity ≥ 0.05
4. **AI Keyword Filtering**: Use 70+ AI domain core keywords for secondary filtering
5. **Result Sorting**: Sort by similarity from high to low, select top 5 resources of each type

### AI Summary Generation

1. **Prioritize OpenAI API**: If API Key is configured, use GPT-3.5-turbo to generate intelligent summaries
2. **Smart Fallback**:
   - Extract Abstract/summary fields
   - Extract first 2-3 meaningful sentences
   - Generate descriptions based on resource type and source
3. **Content Cleaning**: Automatically remove metadata, contact information, and other irrelevant content

## Development Guide

### Extend Search Sources

Add new search functions in `backend/core/resource_searcher.py`:

```python
def search_new_source(keyword: str, max_results: int = 10) -> List[Dict]:
    # Implement new search logic
    pass
```

### Adjust Recommendation Algorithm

Modify similarity calculation methods in `backend/core/recommender.py`:

```python
def compute_similarity(user_docs, resource_content):
    # Custom similarity calculation logic
    pass
```

### Customize Keyword Count

Modify keyword extraction parameters in `app.py`:

```python
keywords = extract_keywords_from_folder(upload_path, top_k=15)  # Modify top_k parameter
```

### Adjust Similarity Threshold

Modify threshold in `backend/core/recommender.py`:

```python
SIMILARITY_THRESHOLD = 0.05  # Modify threshold
```

## FAQ

### Q: What if upload fails?

**A:** Please check:
- Is the file in ZIP format?
- Does the ZIP contain at least 10 TXT/PDF documents?
- Is the file size within the 500MB limit?

### Q: Processing takes too long?

**A:** Processing time depends on:
- Number and size of documents
- Network connection speed (needs to access external resources)
- Usually takes 2-5 minutes

### Q: Recommendation results are not relevant?

**A:** Please ensure:
- Uploaded documents are on AI/Machine Learning topics
- Sufficient number of documents (at least 10)
- High quality document content

### Q: OpenAI API call fails?

**A:** The system will automatically use fallback methods to generate summaries, and all features remain fully functional. If you want to use AI summaries, please check:
- Is the API Key correctly configured?
- Is the network connection normal?
- Is the API quota sufficient?

### Q: Port is occupied?

**A:** Modify the last line in `app.py` to change the port number:

```python
app.run(debug=True, port=5001)
```

## Contributing

We welcome all forms of contributions!

### How to Contribute

1. **Fork** this repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a **Pull Request**

### Contribution Directions

- Fix bugs
- Add new features
- Improve documentation
- UI/UX optimization
- Performance optimization
- Extend search sources
- Add tests

## License

This project is licensed under the [Apache-2.0](LICENSE) License.

## Contact

- **GitHub Issues**: [Submit Issues](https://github.com/LiiiKiii/Reinforcement-Ebook-System/issues)
- **Project Homepage**: [Visit Project](https://github.com/LiiiKiii/Reinforcement-Ebook-System)

## Acknowledgments

Thanks to all developers and users who have contributed to this project!

---

<div align="center">

**If this project helps you, please give it a Star!**

Made with love by [LiiiKiii](https://github.com/LiiiKiii)

</div>
