import os
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------- PDF utils ----------
def extract_text_from_pdf(file_like) -> str:
    """
    file_like can be a path or a file object (Flask/Streamlit uploader provides file-like).
    """
    reader = PdfReader(file_like)
    text_parts = []
    for page in reader.pages:
        text_parts.append((page.extract_text() or ""))
    return "\n".join(text_parts).strip()

# ---------- Refine summarizer ----------
def refine_summarize_text(
    text: str,
    model: str = "gemini-2.0-flash-exp",
    temperature: float = 0.1,
    chunk_size: int = 1800,
    chunk_overlap: int = 200,
) -> str:
    if not text or not text.strip():
        return "No text found to summarize."

    # LLM
    llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    docs = splitter.split_documents([Document(page_content=text)])

    # Prompts for Refine chain
    map_prompt = PromptTemplate.from_template(
        "Write a concise, factual summary of the following text.\n\n{text}\n\nSummary:"
    )
    refine_prompt = PromptTemplate.from_template(
        "You have an existing draft summary:\n\n{existing_answer}\n\n"
        "Refine it using the new context below. Add missing key facts, remove duplication, "
        "and keep it crisp (<= 150 words if possible). If the new context adds nothing, return the existing summary.\n\n"
        "New context:\n{text}\n\nRefined Summary:"
    )

    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=map_prompt,
        refine_prompt=refine_prompt,
        verbose=False
    )

    # Use invoke() for newer langchain versions
    result = chain.invoke({"input_documents": docs})
    return result["output_text"]