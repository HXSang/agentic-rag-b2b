import fitz
import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(pdf_path : str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages=[]
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        if len(text.strip()) < 100:
            continue
        
        pages.append({
            "page_num": page_num+1,
            "text" : text,
            "source" : Path(pdf_path).name
        })
        
    print(f"Loaded {len(pages)} pages from {Path(pdf_path).name}")
    return pages
    
def chunk_text(pages: list):
    chunk_size = 1000
    chunk_overlap = 100
    chunks = []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    
    for page in pages:
        text_content = page["text"]
        page_num = page["page_num"]
        source = page["source"]
        
        page_chunks = splitter.split_text(text_content)
        
        for i, chunk_str in enumerate(page_chunks):
            chunks.append({
                "chunk_id" : f"{source}_p{page_num}_c{i}",
                "text": chunk_str,
                "page_num": page_num,
                "source" : source
            })
    return chunks