# pdf_processor.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def process_pdfs(pdf_directory):
    # Create the pdfs directory if it doesn't exist
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)
        print(f"Created directory: {pdf_directory}")
        print("Please add your PDF files to this directory and run the script again.")
        return []
    
    documents = []
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_directory}")
        print("Please add PDF files to the directory and run the script again.")
        return []
    
    for file in pdf_files:
        pdf_path = os.path.join(pdf_directory, file)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
        print(f"Processed: {file}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    splits = text_splitter.split_documents(documents)
    return splits