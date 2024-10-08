# setup_index.py
from azure_search_manager import AzureSearchManager
from pdf_processor import process_pdfs
import os

def main():
    # Initialize the search manager
    search_manager = AzureSearchManager()
    
    try:
        # Create new index
        print("\n1. Creating search index...")
        search_manager.create_index()
        
        # Process PDFs
        print("\n2. Processing PDF documents...")
        pdf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdfs")
        documents = process_pdfs(pdf_dir)
        
        if not documents:
            print("No documents found to process!")
            return
        
        # Upload documents
        print(f"\n3. Uploading {len(documents)} documents to search index...")
        search_manager.upload_documents(documents)
        
        # Verify the index
        print("\n4. Verifying index setup...")
        search_manager.verify_index()
        
    except Exception as e:
        print(f"\n❌ Error during setup: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Check your .env file contains correct credentials")
        print("2. Verify your Azure Search service is running")
        print("3. Ensure you have admin access to the service")
        print("4. Check if your service quota/limits are not exceeded")

if __name__ == "__main__":
    main()