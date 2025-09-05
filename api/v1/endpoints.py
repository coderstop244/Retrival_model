from fastapi import APIRouter, Depends, HTTPException, Header, status,Request
from models.schemas import HackRxRequest, HackRxResponse
from core.document_loader import load_and_chunk_documents
from core.vectorstore import get_vectorstore, search_clauses
from core.llm import answer_questions
import os
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()

router = APIRouter()


async def verify_auth_token(authorization: Optional[str] = Header(None)):
    """
    Dependency function to verify the Authorization header.
    """
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid.")
    
    token = authorization.split(" ")[1]
    if token != os.getenv("AUTH_KEY"):
        raise HTTPException(status_code=403, detail="Invalid authorization token.")
    return True


@router.post("/hackrx/run", response_model=HackRxResponse)
async def run_query(
    request: HackRxRequest,
    authenticated: bool = Depends(verify_auth_token)
):
    # Auth check
    auth = request.headers.get("Authorization")
    if auth != os.getenv("AUTH_KEY"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid token")
    # 1. Load and chunk documents
    docs = await load_and_chunk_documents(request.documents)
    try:
            # 2. Generate embeddings and upsert to Pinecone
            vectorstore = await get_vectorstore(docs)
            
            if not vectorstore:
                raise HTTPException(status_code=500, detail="Failed to create vector store.")
        # 3. For each question, search and answer
            answers = []
            for question in request.questions:
                print(f"Processing question: {question}")
                answer = await answer_questions(vectorstore, question)
                if not answer:
                    raise HTTPException(status_code=500, detail=f"Failed to get answer for question: {question}")
                answers.append(answer)

            return HackRxResponse(answers=answers)
    
    except HTTPException as e:
        # Re-raise HTTP exceptions to be handled by FastAPI
        raise e



@router.get("/hackrx/health")
async def health_check():
    return {"status": "healthy"}

@router.get("/hackrx/docs")
async def get_docs():
    return {"message": "This endpoint is deprecated. Please use /hackrx/run for answers."}