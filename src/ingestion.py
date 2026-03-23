import fitz
import os
import chromadb
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import re

load_dotenv()

import re

def clean_text(text: str) -> str:
    bullet_pattern = r'^[■▪●○•⁃➢➔\u25a0\u25aa\u2022\uf0a7\uf0fc\uf02d\x02]+\s*'
    text = re.sub(bullet_pattern, '- ', text, flags=re.MULTILINE)
    
    text = re.sub(r'I N N O V A T I O N S\s*\|\s*S I C K', '', text)
    
    text = re.sub(r'\d{7}/\d{4}-\d{2}-\d{2}', '', text)
    
    text = re.sub(r'Subject to change without notice', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'-\s*www\.sick\.com/\S+', '', text)
    text = re.sub(r'For more information, simply enter the link or scan the QR( code)?.*', '', text, flags=re.IGNORECASE)
    
    text = re.sub(r'^\s*\d\s*\d?\s*$', '', text, flags=re.MULTILINE)

    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text) 
    
    return text.strip()

def load_pdf(pdf_path : str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages=[]
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = clean_text(page.get_text())
        
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

def embed_and_store(chunks):
    client = chromadb.PersistentClient(path="vectordb/chroma_store")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name = os.getenv("EMBEDDING_MODEL")
    )
    collection = client.get_or_create_collection(
        name = "belt_catalog",
        embedding_function = embedding_fn
    )
    for chunk in chunks:
        collection.add(
            documents=[chunk["text"]],
            ids=[chunk["chunk_id"]],
            metadatas=[{
                "source": chunk["source"],
                "page_num": chunk["page_num"]
                }]
            )
        
    return collection