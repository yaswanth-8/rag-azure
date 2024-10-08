# azure_credentials_check.py
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from dotenv import load_dotenv
import os

def verify_azure_search_credentials():
    load_dotenv()
    
    # Get credentials from environment
    endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
    key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
    
    # Verify environment variables exist
    if not all([endpoint, key, index_name]):
        print("\nMissing environment variables:")
        if not endpoint:
            print("❌ AZURE_SEARCH_SERVICE_ENDPOINT not found")
        if not key:
            print("❌ AZURE_SEARCH_ADMIN_KEY not found")
        if not index_name:
            print("❌ AZURE_SEARCH_INDEX_NAME not found")
        return False
    
    # Verify endpoint format
    if not endpoint.startswith("https://") or not endpoint.endswith(".search.windows.net"):
        print("\n❌ Invalid endpoint format.")
        print("Expected format: https://your-service-name.search.windows.net")
        print(f"Current value: {endpoint}")
        return False
    
    try:
        # Try to create a client
        credential = AzureKeyCredential(key)
        index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
        
        # Try to list indexes to verify permissions
        print("\nTesting Azure Search permissions...")
        indexes = list(index_client.list_indexes())
        print("✅ Successfully connected to Azure Search")
        print(f"Found {len(indexes)} existing indexes:")
        for idx in indexes:
            print(f"  - {idx.name}")
            
        return True
        
    except Exception as e:
        print("\n❌ Error connecting to Azure Search:")
        if "Forbidden" in str(e):
            print("Permission denied. Please verify your API key has admin permissions.")
            print("\nTo fix this:")
            print("1. Go to Azure Portal")
            print("2. Navigate to your Search service")
            print("3. Go to 'Keys' section")
            print("4. Copy the 'Primary admin key' (not the query key)")
            print("5. Update your .env file with the correct key")
        else:
            print(f"Error details: {str(e)}")
        return False

if __name__ == "__main__":
    print("Verifying Azure Search credentials...")
    verify_azure_search_credentials()