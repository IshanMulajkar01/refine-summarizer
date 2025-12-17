import os
import streamlit as st
from dotenv import load_dotenv
from summarizer import extract_text_from_pdf, refine_summarize_text

load_dotenv()
assert os.getenv("GOOGLE_API_KEY"), "Set GOOGLE_API_KEY in environment!"

st.set_page_config(page_title="Refine Summarizer", layout="wide")
st.title("Refine Summarizer (Gemini + LangChain)")

st.write("Paste text or upload a PDF. The app will use **RefineChain** to iteratively build a concise, accurate summary.")

with st.sidebar:
    model = st.selectbox("Model", ["gemini-2.0-flash-exp", "gemini-1.5-pro"], index=0)
    chunk_size = st.slider("Chunk size", 500, 6000, 1800, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 1000, 200, 50)
    st.caption("Use smaller chunk size if you hit rate limits.")

tab1, tab2 = st.tabs(["Paste Text", "Upload PDF"])

text_input = ""
with tab1:
    text_input = st.text_area("Text", height=240, placeholder="Paste your document here...")

with tab2:
    pdf = st.file_uploader("PDF file", type=["pdf"])

if st.button("Summarize", use_container_width=True):
    text = text_input.strip()
    if not text and pdf is not None:
        try:
            text = extract_text_from_pdf(pdf)
        except Exception as e:
            st.error(f"Could not read PDF: {e}")

    if not text:
        st.warning("Please paste text or upload a PDF.")
    else:
        with st.spinner("Summarizing with RefineChain..."):
            try:
                summary = refine_summarize_text(
                    text=text,
                    model=model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                st.success("Done!")
                st.subheader("Summary")
                st.write(summary)
            except Exception as e:
                st.error(f"Error while summarizing: {e}")