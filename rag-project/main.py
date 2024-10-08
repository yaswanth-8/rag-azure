from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from azure_search_manager import AzureSearchManager
from rag_chain import RAGChain

app = FastAPI(debug=True)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to limit origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

search_manager = AzureSearchManager()
rag_chain = RAGChain()

class Question(BaseModel):
    text: str

@app.post("/ask")
async def ask_question(question: Question):
    try:
        # Search for relevant documents
        context = search_manager.search_documents(question.text)
        
        # Generate response using RAG
        response = rag_chain.generate_response(
            context="\n".join(context),
            question=question.text
        )
        
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
