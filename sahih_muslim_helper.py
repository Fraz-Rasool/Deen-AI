import os
import pandas as pd
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from tqdm import tqdm  



# Load environment variables
load_dotenv()
gem_api = os.getenv("GEMINI_API_KEY")
pinecone_api = os.getenv("PINECONE_API_KEY")

# Constants
CSV_PATH = "Combined Sahih Muslim CSV.csv"
INDEX_NAME = "sahimuslim-index"
DIMENSIONS = 768 # Google embedding size
REGION = "us-east-1"

# Set up LLM and embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5, google_api_key=gem_api)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gem_api)

# Pinecone setup
pc = PineconeClient(api_key=pinecone_api)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSIONS,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=REGION)
    )

# Load and chunk Qur’an CSV
def load_sahi_muslim_csv():
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    documents = []
    for _, row in df.iterrows():
        content = (
    f"**Hadith {row['hadithNumber']}**\n"
    f"**Narrated by:** {row['englishNarrator']}\n"
    f"**Book:** {row['bookName']} | **Chapter:** {row['chapterEnglish']}\n\n"
    f"{row['hadithEnglish']}"
    )

        metadata = {
    "hadithNumber": row["hadithNumber"],
    "englishNarrator": row["englishNarrator"],
    "bookName": row["bookName"],
    "chapterEnglish": row["chapterEnglish"],
    "writerName": row.get("writerName", ""),
    "volume": row.get("volume", ""),
    "status": row.get("status", "")
    }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents  # No chunking needed if ayahs are already concise


# Create vector store using Pinecone
def create_vector_store_sahi_muslim(documents, batch_size=100):
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=pinecone_api
    )

    # Batch upload to avoid exceeding 4MB API limit
    for i in tqdm(range(0, len(documents), batch_size), desc="🔁 Uploading to Pinecone"):
        batch = documents[i:i+batch_size]
        try:
            vector_store.add_documents(batch)
        except Exception as e:
            print(f"❌ Error uploading batch {i}-{i+batch_size}: {e}")

    print(f"\n✅ Successfully uploaded {len(documents)} documents to Pinecone.")

# Load Pinecone vector store
def load_vector_store_sahi_muslim():
    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=pinecone_api
    )

# QA Prompt
def get_conversational_chain_sahi_muslim():
    prompt_template = """You are DeenAI, an Islamic assistant helping users with authentic responses strictly from authentic Hadith sources.

Given the following context (extracted from the Hadith), answer the user's question **strictly based** on the Hadith translations provided.
In your answer:
- Clearly present the most relevant **Hadith**.
- Always include the following reference details:
  - **Hadith Number**
  - **Narrator**
  - **Book Name**
  - **Chapter Title**
  - **Writer Name** (if available)
  - **Volume** (if available)
  - **Authentication Status** (e.g., Sahih, Daif)
- Do **not** provide an answer if the context does not contain a clear Hadith for the query. Instead, respond with: **"I don't know."**

CONTEXT: {context}

QUESTION: {question}

Answer:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

# Handle user query
def user_query_sahi_muslim(query):
    vector_store = load_vector_store_sahi_muslim()
    docs = vector_store.similarity_search(query, k=10)
    chain = get_conversational_chain_sahi_muslim()
    result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    return result["output_text"]