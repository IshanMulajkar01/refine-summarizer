import os
from pypdf import PdfReader
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv

load_dotenv()

def summarize_pdf(pdf_path):
    # Read PDF
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.create_documents([text])
    
    # Setup LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )
    
    # Create prompt template
    prompt_template = """
    Write a concise summary of the following text:
    {text}
    CONCISE SUMMARY:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    
    # Load chain
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=prompt,
        combine_prompt=prompt
    )
    
    # Generate summary
    summary = chain.run(chunks)
    return summary
