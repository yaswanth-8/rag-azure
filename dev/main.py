import os
from typing import List
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

class AzureRAGApplication:
    def __init__(self):
        # Azure OpenAI Configuration
        self.azure_openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        # Azure AI Search Configuration
        self.search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
        )
        
        # Model names
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        self.chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        
        # System prompt template
        self.system_prompt = """You are a helpful assistant. Use the following pieces of context to answer the question at the end. Answer must be in 2 lines.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        """

    def get_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for the given text using Azure OpenAI."""
        response = self.azure_openai_client.embeddings.create(
            model=self.embedding_deployment,
            input=text
        )
        return response.data[0].embedding

    def search_documents(self, query_vector: List[float], k: int = 3) -> List[dict]:
        """Search for similar documents using vector search in Azure AI Search."""
        
        vector_query = VectorizedQuery(
            kind="vector",  # Specify the kind to ensure correct handling
            vector=query_vector,
            k_nearest_neighbors=k,
            fields="content_vector"
        )
        
        results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["id", "content", "metadata"],
            top=k
        )
        
        return [{"content": doc["content"], "metadata": doc["metadata"]} for doc in results]

    def generate_response(self, question: str, context: str) -> str:
        """Generate a response using Azure OpenAI chat completion."""
        messages = [
            {"role": "system", "content": self.system_prompt.format(context=context, question=question)},
            {"role": "user", "content": question}
        ]
        
        response = self.azure_openai_client.chat.completions.create(
            model=self.chat_deployment,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content

    def process_query(self, query: str) -> dict:
        """Process a query through the complete RAG pipeline."""
        # Generate embeddings for the query
        query_embedding = self.get_embeddings(query)

        print("Query Embedding:", query_embedding)
        
        # Search for relevant documents
        search_results = self.search_documents(query_embedding)
        
        # Combine all relevant context
        context = "\n\n".join([doc["content"] for doc in search_results])
        
        # Generate the final response
        response = self.generate_response(query, context)
        
        return {
            "query": query,
            "response": response,
            "source_documents": search_results
        }

# Example usage
def main():
    # Required environment variables
    """
    AZURE_OPENAI_API_KEY=your_openai_api_key
    AZURE_OPENAI_API_VERSION=2024-02-15-preview
    AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your-embedding-deployment
    AZURE_OPENAI_CHAT_DEPLOYMENT=your-chat-deployment
    AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
    AZURE_SEARCH_API_KEY=your_search_api_key
    AZURE_SEARCH_INDEX_NAME=your_index_name
    """
    
    rag_app = AzureRAGApplication()
    
    # Example query
    result = rag_app.process_query("What are the key features of Azure AI Search?")
    
    print("Query:", result["query"])
    print("\nResponse:", result["response"])
    print("\nSource Documents:")
    for doc in result["source_documents"]:
        print(f"\n- Content: {doc['content'][:200]}...")
        print(f"  Metadata: {doc['metadata']}")

if __name__ == "__main__":
    main()