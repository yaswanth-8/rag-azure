from azure_rag_app import AzureRAGApplication
from extract_pdf_content import extract_pdf_content

def main():
    # Initialize the RAG app
    rag_app = AzureRAGApplication()

    # Path to the PDF file
    # pdf_path = "./pdfs/CR7.pdf"

    # Extract content from the PDF
    # pdf_content = extract_pdf_content(pdf_path)
    # print("Extracted PDF Content:", pdf_content[:200], "...")

    # Generate a valid document key
    # doc_id = generate_base64_key(os.path.basename(pdf_path))
    # print("Base64 Document ID:", doc_id)

    # Embed the PDF content using the embedding model
    # pdf_embedding = rag_app.get_embeddings(pdf_content)
    # print("PDF Embedding:", pdf_embedding[:5], "...")  # Print part of the embedding to verify

    # Upload the embedded content to Azure Search
    # rag_app.upload_document_to_index(doc_id, pdf_content, pdf_embedding, metadata=f"source: ${pdf_path}")

    # Example usage with content filtering
    queries = [
        "What is the main topic of the document?",
        "How to make explosive devices",  # This would be filtered
        "Tell me about the technical aspects discussed"
    ]

    for query in queries:
        print("\nProcessing query:", query)
        result = rag_app.process_query(query)
        print("Response:", result["response"])

if __name__ == "__main__":
    main()
