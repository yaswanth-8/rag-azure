# init.py
from pdf_processor import process_pdfs
from azure_search_manager import AzureSearchManager
import os

def main():
    # Ensure we're in the correct directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = os.path.join(project_root, "pdfs")
    
    # Process PDFs
    print("Processing PDFs...")
    documents = process_pdfs(pdf_dir)
    
    if not documents:
        print("No documents to process. Please add PDFs and try again.")
        return
    
    # Upload to Azure Search
    print("Uploading to Azure Search...")
    search_manager = AzureSearchManager()
    search_manager.create_index()
    search_manager.upload_documents(documents)
    print("Upload complete!")

if __name__ == "__main__":
    main()