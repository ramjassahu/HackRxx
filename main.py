# main.py

from fastapi import FastAPI, Header, HTTPException, Security
from pydantic import BaseModel, HttpUrl
from typing import List, Annotated
import os
import requests
import tempfile
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredEmailLoader,
)
from dotenv import load_dotenv
import asyncio

load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_groq import ChatGroq

llm = ChatGroq(
    api_key=os.getenv("GROK_API_KEY"),
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
)


embedding = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    user_agent="langchain",
)


def get_loader_from_url(url: str):
    url_str = str(url)  # Convert HttpUrl to string for manipulation
    file_ext = url_str.split("?")[0].split(".")[-1].lower()

    response = requests.get(url_str)
    if not response.ok:
        raise ValueError(f"Failed to download file: {url_str}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}")
    tmp.write(response.content)
    tmp.close()

    if file_ext == "pdf":
        return PyPDFLoader(tmp.name), tmp.name
    elif file_ext in ["doc", "docx"]:
        return UnstructuredWordDocumentLoader(tmp.name), tmp.name
    elif file_ext in ["eml", "msg"]:
        return UnstructuredEmailLoader(tmp.name), tmp.name
    else:
        raise ValueError(f"Unsupported file type: .{file_ext}")


# ---- MAIN PIPELINE ----
def build_retriever_from_url(url):
    loader, temp_path = get_loader_from_url(url)
    pages = loader.load()
    full_text = "\n".join([page.page_content for page in pages])

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    splits = splitter.create_documents([full_text])

    # Build FAISS and BM25
    vectordb = FAISS.from_documents(splits, embedding=embedding)
    faiss_retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 3

    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )

    return ensemble_retriever, temp_path


async def process_single_query(ensemble_retriever, llm, query: str) -> str:
    """
    Asynchronously retrieves documents and generates an answer for a single query.
    """
    # Use the asynchronous 'ainvoke' method for non-blocking calls
    docs = await ensemble_retriever.ainvoke(query)
    relevant_docs = "\n".join([doc.page_content for doc in docs])

    prompt = f"Answer the following question in 1 line and concisely based on the provided documents:\n\nQuestion: {query}\n\nDocuments:\n{relevant_docs}\n\nAnswer:"

    # Use the asynchronous 'ainvoke' for the language model call
    answer = await llm.ainvoke(prompt)
    return answer.content


# --- Configuration ---
# In a real application, this key would be stored securely,
# for example, in an environment variable.
API_KEY = "50255bc08e08f6431861bace6cb7232a1a9c317758fc6afd177cfd01ba3cff9a"
BEARER_TOKEN = f"Bearer {API_KEY}"

# --- Pydantic Models for Data Validation ---


# This model defines the expected structure of the JSON body in the request.
# Pydantic automatically validates the incoming data against this structure.
class HackRxRequest(BaseModel):
    documents: HttpUrl  # Validates that 'documents' is a valid URL.
    questions: List[str]  # Validates that 'questions' is a list of strings.


# --- FastAPI Application ---

app = FastAPI(
    title="HackRx API Processor",
    description="A simple API to receive and process documents and questions.",
    version="1.0.0",
)

# --- Security Dependency ---


# This function checks for the Authorization header.
async def get_api_key(authorization: Annotated[str | None, Header()] = None):
    """
    Dependency to verify the bearer token in the Authorization header.
    """
    if authorization is None:
        raise HTTPException(status_code=401, detail="Authorization header is missing")
    if authorization != BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid API Key")


# --- API Endpoint ---


@app.post("/api/v1/hackrx/run", summary="Process a document and questions")
async def handle_hackrx_request(
    request_data: HackRxRequest,
    # The Security function injects the dependency and handles errors.
    # The endpoint code will only run if get_api_key succeeds.
    api_key: str = Security(get_api_key),
):
    """
    This endpoint receives a URL to a document and a list of questions.

    It validates the structure of the incoming data and the API key.
    For this basic example, it simply returns a confirmation.

    In a real-world scenario, you would add logic here to:
    1. Download the PDF from the request_data.documents URL.
    2. Extract text from the PDF.
    3. Use a language model or other logic to answer each question in request_data.questions.
    4. Return the answers in the response.
    """
    # print("--- Request Received ---")
    print(f"Document URL: {request_data.documents}")
    print(f"Number of Questions: {len(request_data.questions)}")
    print(request_data.questions);
    document_url = request_data.documents

    ensemble_retriever, temp_file_path = build_retriever_from_url(document_url)

    queries = request_data.questions

    tasks = [process_single_query(ensemble_retriever, llm, query) for query in queries]

    # --- NEW: Execute all tasks concurrently ---
    # asyncio.gather runs all the tasks at the same time and waits for all of them to complete.
    # The '*' unpacks the list of tasks into arguments for gather.
    print("--- Sending all questions to LLM in parallel ---")
    answers = await asyncio.gather(*tasks)

    # Cleanup the temporary file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    print("--- All answers received, sending response ---")
    return answers


# To run this application:hello
# 1. Make sure you have fastapi and uvicorn installed:
#    pip install fastapi "uvicorn[standard]"
# 2. Save this code as a file (e.g., main.py).
# 3. Run the server from your terminal:
#    uvicorn main:app --reload
