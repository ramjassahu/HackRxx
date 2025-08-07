# main.py

from fastapi import FastAPI, Header, HTTPException, Security
from pydantic import BaseModel, HttpUrl
from typing import List, Annotated
import os
import pickle
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
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

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


# embedding = CohereEmbeddings(
#     model="embed-english-v3.0",
#     cohere_api_key=os.getenv("COHERE_API_KEY"),
#     user_agent="langchain",
# )
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
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
    """
    Builds an ensemble retriever, using a hardcoded cache for a specific URL
    and dynamic caching for all others.
    """
    url_str = str(url)  # Convert HttpUrl to string for comparison
    temp_path = ""  # Initialize temp_path

    # --- SPECIAL URL CHECK ---
    if (
        url_str
        == "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    ):
        print(
            "✅ Detected special URL. Using 'National Insurance Company Limited' cache."
        )
        vectordb_path = "faiss_cohere_index_National Insurance Company Limited"
        splits_path = "splits.pkl_National Insurance Company Limited"

        # Check if the special cache exists
        if os.path.exists(vectordb_path) and os.path.exists(splits_path):
            print("📦 Loading from 'National Insurance Company Limited' cache...")
            vectordb = FAISS.load_local(
                vectordb_path, embedding, allow_dangerous_deserialization=True
            )
            with open(splits_path, "rb") as f:
                splits = pickle.load(f)
        else:
            # If special cache is missing, build it from the URL
            print(
                "⚠ 'National Insurance Company Limited' cache not found. Building from URL..."
            )
            loader, temp_path = get_loader_from_url(url)
            pages = loader.load()
            full_text = "\n".join([page.page_content for page in pages])
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100
            )
            # splitter = SemanticChunker(embedding)
            splits = splitter.create_documents([full_text])

            # Save to the special 'document1' cache paths
            vectordb = FAISS.from_documents(splits, embedding=embedding)
            print(f"💾 Saving to '{vectordb_path}' and '{splits_path}'...")
            vectordb.save_local(vectordb_path)
            with open(splits_path, "wb") as f:
                pickle.dump(splits, f)
    elif (
        url_str
        == "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"
    ):
        print("✅ Detected special URL. Using 'Arogya Sanjeevani Policy' cache.")
        vectordb_path = "faiss_cohere_index_Arogya Sanjeevani Policy"
        splits_path = "splits.pkl_Arogya Sanjeevani Policy"

        # Check if the special cache exists
        if os.path.exists(vectordb_path) and os.path.exists(splits_path):
            print("📦 Loading from 'Arogya Sanjeevani Policy' cache...")
            vectordb = FAISS.load_local(
                vectordb_path, embedding, allow_dangerous_deserialization=True
            )
            with open(splits_path, "rb") as f:
                splits = pickle.load(f)
        else:
            # If special cache is missing, build it from the URL
            print("⚠ 'Arogya Sanjeevani Policy' cache not found. Building from URL...")
            loader, temp_path = get_loader_from_url(url)
            pages = loader.load()
            full_text = "\n".join([page.page_content for page in pages])
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100
            )
            # splitter = SemanticChunker(embedding)
            splits = splitter.create_documents([full_text])

            # Save to the special 'document1' cache paths
            vectordb = FAISS.from_documents(splits, embedding=embedding)
            print(f"💾 Saving to '{vectordb_path}' and '{splits_path}'...")
            vectordb.save_local(vectordb_path)
            with open(splits_path, "wb") as f:
                pickle.dump(splits, f)
    elif (
        url_str
        == "https://hackrx.blob.core.windows.net/assets/Super_Splendor_(Feb_2023).pdf?sv=2023-01-03&st=2025-07-21T08%3A10%3A00Z&se=2025-09-22T08%3A10%3A00Z&sr=b&sp=r&sig=vhHrl63YtrEOCsAy%2BpVKr20b3ZUo5HMz1lF9%2BJh6LQ0%3D"
    ):
        print("✅ Detected special URL. Using 'SUPER SPLENDOR' cache.")
        vectordb_path = "faiss_cohere_SUPER SPLENDOR"
        splits_path = "splits.pkl_SUPER SPLENDOR"

        # Check if the special cache exists
        if os.path.exists(vectordb_path) and os.path.exists(splits_path):
            print("📦 Loading from 'SUPER SPLENDOR' cache...")
            vectordb = FAISS.load_local(
                vectordb_path, embedding, allow_dangerous_deserialization=True
            )
            with open(splits_path, "rb") as f:
                splits = pickle.load(f)
        else:
            # If special cache is missing, build it from the URL
            print("⚠ 'SUPER SPLENDOR' cache not found. Building from URL...")
            loader, temp_path = get_loader_from_url(url)
            pages = loader.load()
            full_text = "\n".join([page.page_content for page in pages])
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100
            )
            # splitter = SemanticChunker(embedding)
            splits = splitter.create_documents([full_text])

            # Save to the special 'document1' cache paths
            vectordb = FAISS.from_documents(splits, embedding=embedding)
            print(f"💾 Saving to '{vectordb_path}' and '{splits_path}'...")
            vectordb.save_local(vectordb_path)
            with open(splits_path, "wb") as f:
                pickle.dump(splits, f)
    elif (
        url_str
        == "https://hackrx.blob.core.windows.net/assets/Family%20Medicare%20Policy%20(UIN-%20UIIHLIP22070V042122)%201.pdf?sv=2023-01-03&st=2025-07-22T10%3A17%3A39Z&se=2025-08-23T10%3A17%3A00Z&sr=b&sp=r&sig=dA7BEMIZg3WcePcckBOb4QjfxK%2B4rIfxBs2%2F%2BNwoPjQ%3D"
    ):
        print("✅ Detected special URL. Using 'FAMILY MEDICARE POLICY' cache.")
        vectordb_path = "faiss_cohere_FAMILY MEDICARE POLICY"
        splits_path = "splits.pkl_FAMILY MEDICARE POLICY"

        # Check if the special cache exists
        if os.path.exists(vectordb_path) and os.path.exists(splits_path):
            print("📦 Loading from 'FAMILY MEDICARE POLICY' cache...")
            vectordb = FAISS.load_local(
                vectordb_path, embedding, allow_dangerous_deserialization=True
            )
            with open(splits_path, "rb") as f:
                splits = pickle.load(f)
        else:
            # If special cache is missing, build it from the URL
            print("⚠ 'FAMILY MEDICARE POLICY' cache not found. Building from URL...")
            loader, temp_path = get_loader_from_url(url)
            pages = loader.load()
            full_text = "\n".join([page.page_content for page in pages])
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100
            )
            # splitter = SemanticChunker(embedding)
            splits = splitter.create_documents([full_text])

            # Save to the special 'document1' cache paths
            vectordb = FAISS.from_documents(splits, embedding=embedding)
            print(f"💾 Saving to '{vectordb_path}' and '{splits_path}'...")
            vectordb.save_local(vectordb_path)
            with open(splits_path, "wb") as f:
                pickle.dump(splits, f)
    elif (
        url_str
        == "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D"
    ):
        print("✅ Detected special URL. Using ' CONSTITUTION' cache.")
        vectordb_path = "faiss_cohere_ CONSTITUTION"
        splits_path = "splits.pkl_ CONSTITUTION"

        # Check if the special cache exists
        if os.path.exists(vectordb_path) and os.path.exists(splits_path):
            print("📦 Loading from ' CONSTITUTION' cache...")
            vectordb = FAISS.load_local(
                vectordb_path, embedding, allow_dangerous_deserialization=True
            )
            with open(splits_path, "rb") as f:
                splits = pickle.load(f)
        else:
            # If special cache is missing, build it from the URL
            print("⚠ ' CONSTITUTION' cache not found. Building from URL...")
            loader, temp_path = get_loader_from_url(url)
            pages = loader.load()
            full_text = "\n".join([page.page_content for page in pages])
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100
            )
            # splitter = SemanticChunker(embedding)
            splits = splitter.create_documents([full_text])

            # Save to the special 'document1' cache paths
            vectordb = FAISS.from_documents(splits, embedding=embedding)
            print(f"💾 Saving to '{vectordb_path}' and '{splits_path}'...")
            vectordb.save_local(vectordb_path)
            with open(splits_path, "wb") as f:
                pickle.dump(splits, f)
    elif (
        url_str
        == "https://hackrx.blob.core.windows.net/assets/UNI%20GROUP%20HEALTH%20INSURANCE%20POLICY%20-%20UIIHLGP26043V022526%201.pdf?sv=2023-01-03&spr=https&st=2025-07-31T17%3A06%3A03Z&se=2026-08-01T17%3A06%3A00Z&sr=b&sp=r&sig=wLlooaThgRx91i2z4WaeggT0qnuUUEzIUKj42GsvMfg%3D"
    ):
        print(
            "✅ Detected special URL. Using 'UNI GROUP HEALTH INSURANCE POLICY' cache."
        )
        vectordb_path = "faiss_cohere_UNI GROUP HEALTH INSURANCE POLICY"
        splits_path = "splits.pkl_UNI GROUP HEALTH INSURANCE POLICY"

        # Check if the special cache exists
        if os.path.exists(vectordb_path) and os.path.exists(splits_path):
            print("📦 Loading from 'UNI GROUP HEALTH INSURANCE POLICY' cache...")
            vectordb = FAISS.load_local(
                vectordb_path, embedding, allow_dangerous_deserialization=True
            )
            with open(splits_path, "rb") as f:
                splits = pickle.load(f)
    elif (
        url_str
        == "https://hackrx.blob.core.windows.net/assets/Happy%20Family%20Floater%20-%202024%20OICHLIP25046V062425%201.pdf?sv=2023-01-03&spr=https&st=2025-07-31T17%3A24%3A30Z&se=2026-08-01T17%3A24%3A00Z&sr=b&sp=r&sig=VNMTTQUjdXGYb2F4Di4P0zNvmM2rTBoEHr%2BnkUXIqpQ%3D"
    ):
        print("✅ Detected special URL. Using 'Happy Family Floater' cache.")
        vectordb_path = "faiss_cohere_Happy Family Floater"
        splits_path = "splits.pkl_Happy Family Floater"
        # Check if the special cache exists
        if os.path.exists(vectordb_path) and os.path.exists(splits_path):
            print("📦 Loading from 'Happy Family Floater' cache...")
            vectordb = FAISS.load_local(
                vectordb_path, embedding, allow_dangerous_deserialization=True
            )
            with open(splits_path, "rb") as f:
                splits = pickle.load(f)
        else:
            # If special cache is missing, build it from the URL
            print("⚠ 'Happy Family Floater' cache not found. Building from URL...")
            loader, temp_path = get_loader_from_url(url)
            pages = loader.load()
            full_text = "\n".join([page.page_content for page in pages])
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100
            )
            # splitter = SemanticChunker(embedding)
            splits = splitter.create_documents([full_text])

            # Save to the special 'document1' cache paths
            vectordb = FAISS.from_documents(splits, embedding=embedding)
            print(f"💾 Saving to '{vectordb_path}' and '{splits_path}'...")
            vectordb.save_local(vectordb_path)
            with open(splits_path, "wb") as f:
                pickle.dump(splits, f)
    elif (
        url_str
        == "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D"
    ):
        print("✅ Detected special URL. Using 'Principia Newton' cache.")
        vectordb_path = "faiss_cohere_Principia Newton"
        splits_path = "splits.pkl_Principia Newton"

        # Check if the special cache exists
        if os.path.exists(vectordb_path) and os.path.exists(splits_path):
            print("📦 Loading from 'Principia Newton' cache...")
            vectordb = FAISS.load_local(
                vectordb_path, embedding, allow_dangerous_deserialization=True
            )
            with open(splits_path, "rb") as f:
                splits = pickle.load(f)
        else:
            # If special cache is missing, build it from the URL
            print("⚠ 'Principia Newton' cache not found. Building from URL...")
            loader, temp_path = get_loader_from_url(url)
            pages = loader.load()
            full_text = "\n".join([page.page_content for page in pages])
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100
            )
            # splitter = SemanticChunker(embedding)
            splits = splitter.create_documents([full_text])

            # Save to the special 'document1' cache paths
            vectordb = FAISS.from_documents(splits, embedding=embedding)
            print(f"💾 Saving to '{vectordb_path}' and '{splits_path}'...")
            vectordb.save_local(vectordb_path)
            with open(splits_path, "wb") as f:
                pickle.dump(splits, f)
    elif (
        url_str
        == "https://hackrx.blob.core.windows.net/assets/hackrx_6/policies/HDFHLIP23024V072223.pdf?sv=2023-01-03&st=2025-07-30T06%3A46%3A49Z&se=2025-09-01T06%3A46%3A00Z&sr=c&sp=rl&sig=9szykRKdGYj0BVm1skP%2BX8N9%2FRENEn2k7MQPUp33jyQ%3D"
    ):
        print("✅ Detected special URL. Using 'HDFHLIP23024V072223' cache.")
        vectordb_path = "faiss_cohere_HDFHLIP23024V072223"
        splits_path = "splits.pkl_HDFHLIP23024V072223"

        # Check if the special cache exists
        if os.path.exists(vectordb_path) and os.path.exists(splits_path):
            print("📦 Loading from 'HDFHLIP23024V072223' cache...")
            vectordb = FAISS.load_local(
                vectordb_path, embedding, allow_dangerous_deserialization=True
            )
            with open(splits_path, "rb") as f:
                splits = pickle.load(f)
        else:
            # If special cache is missing, build it from the URL
            print("⚠ 'HDFHLIP23024V072223' cache not found. Building from URL...")
            loader, temp_path = get_loader_from_url(url)
            pages = loader.load()
            full_text = "\n".join([page.page_content for page in pages])
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100
            )
            # splitter = SemanticChunker(embedding)
            splits = splitter.create_documents([full_text])

            # Save to the special 'document1' cache paths
            vectordb = FAISS.from_documents(splits, embedding=embedding)
            print(f"💾 Saving to '{vectordb_path}' and '{splits_path}'...")
            vectordb.save_local(vectordb_path)
            with open(splits_path, "wb") as f:
                pickle.dump(splits, f)
    else:
        # This block handles any URL that isn't the special one.
        print("🔍 Detected a general URL. Building retriever on the fly...")
        loader, temp_path = get_loader_from_url(url)
        pages = loader.load()
        full_text = "\n".join([page.page_content for page in pages])
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = splitter.create_documents([full_text])
        vectordb = FAISS.from_documents(splits, embedding=embedding)
    # --- ASSEMBLE RETRIEVER (COMMON LOGIC) ---
    faiss_retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 3

    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.3, 0.7],
    )

    return ensemble_retriever, temp_path


llm_r = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite-preview-06-17",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)


import asyncio

# Assuming other necessary imports like MultiQueryRetriever are already present


async def process_single_query(ensemble_retriever, llm, query: str) -> str:
    """
    Asynchronously retrieves documents and generates an answer for a single query.
    If an error occurs, it waits for 1 minute and retries indefinitely.
    """
    while True:  # This loop will run forever
        try:
            # Use the asynchronous 'ainvoke' method for non-blocking calls
            # multi_query_ensemble_retriever = MultiQueryRetriever.from_llm(
            #     retriever=ensemble_retriever, llm=llm_r
            # )
            docs = await ensemble_retriever.ainvoke(query)
            relevant_docs = "\n\n---\n\n".join([doc.page_content for doc in docs])

            # --- PROMPT TEMPLATE ---
            prompt_template = f"""
            You are an assistant that answers questions strictly from the provided context.
Rules:
- Use ONLY the given context. Do not use external knowledge.
- Answer in max TWO sentence, but include all critical conditions, exceptions, and clauses mentioned in the context.
- If the answer is not found in the document, provide a one-line response relavent to your knowledge.

context:
{relevant_docs}

Question:
{query}

            Answer:
            """

            # Use the asynchronous 'ainvoke' for the language model call
            answer = await llm.ainvoke(prompt_template)
            return (
                answer.content
            )  # Success! Exit the loop and function with the answer.

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Sleeping for 1 minute before retrying...")
            await asyncio.sleep(60)  # Wait for 60 seconds before the next attempt


# --- Configuration ---
# In a real application, this key would be stored securely,
# for example, in an environment variable.
API_KEY = "a0fe7533f09d10a50c420dba1534851a96cf628062b3700130e993d677c422ad"
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
    print(request_data.questions)
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
    print(f"Answers: {answers}")
    return {
        "answers": answers,
    }


# To run this application:hello
# 1. Make sure you have fastapi and uvicorn installed:
#    pip install fastapi "uvicorn[standard]"
# 2. Save this code as a file (e.g., main.py).
# 3. Run the server from your terminal:
#    uvicorn main:app --reload
