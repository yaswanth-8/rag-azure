# azure_search_manager.py
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
)
from azure.core.exceptions import ResourceNotFoundError
import os
from dotenv import load_dotenv
import time

class AzureSearchManager:
    def __init__(self):
        load_dotenv()
        self.endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
        self.key = os.getenv("AZURE_SEARCH_ADMIN_KEY")
        self.index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "documents-index")
        self.credential = AzureKeyCredential(self.key)
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=self.credential
        )
        
    def delete_index(self):
        """Delete the index if it exists"""
        try:
            self.index_client.delete_index(self.index_name)
            print(f"Deleted existing index: {self.index_name}")
            # Wait a moment to ensure the index is fully deleted
            time.sleep(2)
        except ResourceNotFoundError:
            print(f"Index {self.index_name} does not exist")
        except Exception as e:
            print(f"Error deleting index: {str(e)}")
            raise

    def create_index(self):
        """Create a new search index"""
        try:
            # First, delete existing index if it exists
            self.delete_index()
            
            # Define the index fields
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(
                    name="content", 
                    type=SearchFieldDataType.String,
                    analyzer_name="standard.lucene"
                ),
                SimpleField(
                    name="source", 
                    type=SearchFieldDataType.String, 
                    filterable=True
                )
            ]
            
            # Create the index
            index = SearchIndex(
                name=self.index_name,
                fields=fields
            )
            
            result = self.index_client.create_index(index)
            print(f"Created index: {result.name}")
            # Wait a moment to ensure the index is fully created
            time.sleep(2)
            return result
            
        except Exception as e:
            print(f"Error creating index: {str(e)}")
            raise

    def upload_documents(self, documents):
        """Upload documents to the search index"""
        try:
            # Create a search client
            search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=self.credential
            )
            
            # Prepare documents for upload
            docs_to_upload = []
            for i, doc in enumerate(documents):
                docs_to_upload.append({
                    'id': str(i),
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', 'unknown')
                })
                
            # Upload in batches of 1000 (Azure Search limit)
            batch_size = 1000
            for i in range(0, len(docs_to_upload), batch_size):
                batch = docs_to_upload[i:i + batch_size]
                try:
                    results = search_client.upload_documents(documents=batch)
                    success_count = sum(1 for r in results if r.succeeded)
                    print(f"Uploaded batch {i//batch_size + 1}: {success_count}/{len(batch)} succeeded")
                except Exception as e:
                    print(f"Error uploading batch {i//batch_size + 1}: {str(e)}")
                    raise
                
            print(f"Upload completed. Total documents: {len(docs_to_upload)}")
            
        except Exception as e:
            print(f"Error in upload_documents: {str(e)}")
            raise

    def verify_index(self):
        """Verify the index exists and is properly configured"""
        try:
            index = self.index_client.get_index(self.index_name)
            print(f"\nIndex verification:")
            print(f"Name: {index.name}")
            print("Fields:")
            for field in index.fields:
                print(f"  - {field.name} ({field.type})")
            
            # Get document count
            search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=self.credential
            )
            doc_count = search_client.search("*", top=1, include_total_count=True).get_count()
            print(f"Total documents: {doc_count}")
            
            return True
        except Exception as e:
            print(f"Error verifying index: {str(e)}")
            return False
        
    def search_documents(self, query, top=3):
        search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential
        )
        
        results = search_client.search(query, top=top)
        return [result['content'] for result in results]