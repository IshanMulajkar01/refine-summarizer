# Refine Summarizer

AI-powered text and PDF summarizer using Google's Gemini models with LangChain's Refine Chain strategy.

## Features

- 📄 Summarize text or PDF documents
- 🔄 Iterative refinement for higher quality summaries
- 🚀 Flask and Streamlit interfaces
- ⚡ Fast processing with Gemini 2.0 Flash
- 🎯 Customizable chunk sizes and overlap

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/IshanMulajkar01/refine-summarizer.git
cd refine-summarizer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the root directory:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get your API key from: https://aistudio.google.com/app/apikey

## Usage

### Flask Web App
```bash
python app.py
```
Visit: http://localhost:5000

### Streamlit App
```bash
streamlit run streamlit_app.py
```

## How It Works

1. **Extract**: Reads text from input or PDF
2. **Split**: Divides text into overlapping chunks
3. **Refine**: Iteratively summarizes and refines using LangChain's Refine Chain
4. **Output**: Returns a concise, factual summary

## Technologies

- **LangChain**: Framework for LLM applications
- **Google Gemini**: State-of-the-art language models
- **Flask**: Lightweight web framework
- **Streamlit**: Interactive data app framework
- **PyPDF**: PDF text extraction

## License

MIT