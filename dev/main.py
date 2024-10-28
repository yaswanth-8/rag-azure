import os
from typing import List
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI
from langchain_community.document_loaders import PyPDFLoader
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions
from azure.core.exceptions import HttpResponseError

# Load environment variables
load_dotenv()

import base64

def generate_base64_key(filename: str) -> str:
    """Generate a valid Base64-encoded document key."""
    encoded_bytes = base64.urlsafe_b64encode(filename.encode("utf-8"))
    return encoded_bytes.decode("utf-8").rstrip("=")  # Remove padding '=' for Azure compatibility


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
            "content_vector": embedding,  # Upload the embedding vector
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

        # Final safety check on generated response
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
    
class ContentSafetyFilter:
    def __init__(self):
        self.content_safety_client = ContentSafetyClient(
            endpoint=os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_CONTENT_SAFETY_KEY"))
        )

    def is_safe_content(self, text: str) -> tuple[bool, str]:
        """
        Check if the content is safe using Azure Content Safety API.
        Returns a tuple of (is_safe, reason)
        """
        try:
            # Create request
            request = AnalyzeTextOptions(text=text)

            # Analyze text
            response = self.content_safety_client.analyze_text(request)
            
            # Check categories (violence, self-harm, sexual, hate)
            categories = {
                "Violence": response["categoriesAnalysis"][3]["severity"],
                "SelfHarm": response["categoriesAnalysis"][1]["severity"],
                "Sexual": response["categoriesAnalysis"][2]["severity"],
                "Hate": response["categoriesAnalysis"][0]["severity"]
            }

            # Define threshold (2 = Low, 3 = Medium, 4 = High)
            SEVERITY_THRESHOLD = 3
            
            for category, severity in categories.items():
                if severity >= SEVERITY_THRESHOLD:
                    return False, f"Content filtered due to {category} content"
            

            return True, "Content is safe"
            
        except HttpResponseError as e:
            return False, f"Error analyzing content: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}" 

def extract_pdf_content(pdf_path: str) -> str:
    """Extract content from a PDF file using PyPDFLoader."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Combine the content of all pages into one string
    pdf_content = "\n".join([doc.page_content for doc in documents])
    return pdf_content


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
