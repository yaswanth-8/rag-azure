import os
from typing import List
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
from content_safety_filter import ContentSafetyFilter  

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

        self.content_filter = ContentSafetyFilter()

        # System prompt template
        self.system_prompt = """You are a helpful assistant. Use the following pieces of context to answer the question at the end.
        Answer must be in 2 lines. If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        
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
            kind="vector",
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

    def upload_document_to_index(self, doc_id: str, content: str, embedding: List[float], metadata: dict = None):
        """Upload the embedded content to the Azure Search index."""
        print("entered uploading.....")
        if metadata is None:
            metadata = {}

        print("metadata:", metadata)

        document = {
            "id": doc_id,
            "content": content,
            "content_vector": embedding,
            "metadata": metadata
        }

        print("document:", document)

        self.search_client.upload_documents(documents=[document])
        print(f"Document {doc_id} uploaded successfully.")

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
        # Check if query is safe
        is_safe, reason = self.content_filter.is_safe_content(query)
        if not is_safe:
            return {
                "query": query,
                "response": "I apologize, but I cannot process this query as it contains potentially harmful content.",
                "reason": reason,
                "source_documents": []
            }

        query_embedding = self.get_embeddings(query)
        search_results = self.search_documents(query_embedding)

        all_content = "\n".join([doc["content"] for doc in search_results])
        content_safe, content_reason = self.content_filter.is_safe_content(all_content)
        if not content_safe:
            return {
                "query": query,
                "response": "I apologize, but I cannot provide an answer as the relevant content has been filtered.",
                "reason": content_reason,
                "source_documents": []
            }

        context = "\n\n".join([doc["content"] for doc in search_results])
        response = self.generate_response(query, context)

        response_safe, response_reason = self.content_filter.is_safe_content(response)
        if not response_safe:
            return {
                "query": query,
                "response": "I apologize, but I cannot provide the generated response as it contains potentially harmful content.",
                "reason": response_reason,
                "source_documents": []
            }

        return {
            "query": query,
            "response": response,
            "source_documents": search_results
        }
