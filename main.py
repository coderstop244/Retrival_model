from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import shutil

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("HACKRX_API_KEY")  # Set this in your .env

app = FastAPI()

def authenticate(x_api_key: Optional[str]):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key.")

@app.post("/hackrx/run")
async def hackrx_run(
    questions: List[str],
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None)
):
    authenticate(x_api_key)

    # Save uploaded PDF
    file_location = f"documents/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Parse PDF robustly
    try:
        loader = PyPDFLoader(file_location)
        docs = loader.load()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF parsing failed: {str(e)}")

    # Chunk documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    documents = text_splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # Pinecone index (assumes already created)
    index_name = "retrivalai"
    vectorstore = Pinecone.from_documents(
        documents,
        embedding=embeddings,
        index_name=index_name
    )

    # QA (replace with your actual QA logic)
    answers = []
    for q in questions:
        # Dummy answer for illustration; replace with your QA chain
        answers.append(f"Answer to: {q}")

    return JSONResponse(content={"answers": answers})