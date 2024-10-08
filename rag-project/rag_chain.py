# rag_chain.py
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

class RAGChain:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        
        self.prompt_template = ChatPromptTemplate.from_template(
            """Answer the question based on the following context:
            
            Context: {context}
            
            Question: {question}
            
            Answer: """
        )
        
    def generate_response(self, context, question):
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        response = self.llm.predict(prompt)
        return response