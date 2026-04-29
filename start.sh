#!/bin/bash

echo "=========================================="
echo "  AI-Pedia"
echo "=========================================="
echo ""

if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 was not found. Please install Python3 first."
    exit 1
fi

echo "Checking dependencies..."
python3 -c "import flask, numpy, sklearn, requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Warning: some dependencies may be missing."
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: dependency installation failed. Please run: pip3 install -r requirements.txt"
        exit 1
    fi
fi

echo "Creating required directories..."
mkdir -p data/uploads data/results data/outputs

export OPENAI_API_KEY="PUT YOUR OPENAI API KEY HERE"

echo ""
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Note: OPENAI_API_KEY was not detected."
    echo "  - Set an API key to enable optional LLM summaries."
    echo "  - Option 1: run in the terminal: export OPENAI_API_KEY='your-key-here'"
    echo "  - Option 2: edit this script and add: export OPENAI_API_KEY='your-key-here'"
    echo "  - Without an API key, the system will use the rule-based fallback summary."
    echo ""
else
    echo "✓ OPENAI_API_KEY detected. LLM summaries are enabled."
    echo ""
fi

echo "The system uses general web search and does not require any other API keys."
echo "  Text search: Wikipedia / Google Scholar / arXiv and other academic sources"
echo "  Video search: YouTube HTML parsing"
echo "  Image search: Google Images / Bing Images / Unsplash / Pexels"
echo ""

echo "Starting the application..."
echo "Open: http://localhost:5000"
echo "Press Ctrl+C to stop the service"
echo ""

env OPENAI_API_KEY="$OPENAI_API_KEY" python3 app.py
