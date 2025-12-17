import os
from pypdf import PdfReader
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def summarize_text(text, api_key=None):
    """Summarize text using Google Gemini with refine chain."""
    try:
        # Use provided API key or get from environment
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        
        # Initialize the LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.3,
            google_api_key=api_key or os.getenv("GOOGLE_API_KEY")
        )
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=500
        )
        
        # Create documents
        chunks = text_splitter.split_text(text)
        docs = [Document(page_content=chunk) for chunk in chunks]
        
        # Load refine chain
        chain = load_summarize_chain(
            llm=llm,
            chain_type="refine",
            verbose=False
        )
        
        # Generate summary
        summary = chain.run(docs)
        return summary
        
    except Exception as e:
        raise Exception(f"Error during summarization: {str(e)}")

def summarize_pdf(pdf_path, api_key=None):
    """Extract text from PDF and summarize it."""
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        raise Exception("No text found in PDF")
    
    summary = summarize_text(text, api_key)
    return summary

if __name__ == "__main__":
    # Example usage
    pdf_file = "example.pdf"
    if os.path.exists(pdf_file):
        summary = summarize_pdf(pdf_file)
        print("Summary:")
        print(summary)
    else:
        print(f"PDF file '{pdf_file}' not found.")