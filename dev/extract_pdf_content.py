from langchain_community.document_loaders import PyPDFLoader

def extract_pdf_content(pdf_path: str) -> str:
    """Extract content from a PDF file using PyPDFLoader."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Combine the content of all pages into one string
    pdf_content = "\n".join([doc.page_content for doc in documents])
    return pdf_content