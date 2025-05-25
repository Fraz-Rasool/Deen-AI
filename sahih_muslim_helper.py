import os
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
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
DIMENSIONS = 768  # Google embedding size
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

# Load and convert CSV into Documents
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
    return documents

# Create Pinecone vector store
def create_vector_store_sahi_muslim(documents, batch_size=100):
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=pinecone_api
    )

    for i in tqdm(range(0, len(documents), batch_size), desc="🔁 Uploading to Pinecone"):
        batch = documents[i:i+batch_size]
        try:
            vector_store.add_documents(batch)
        except Exception as e:
            print(f"❌ Error uploading batch {i}-{i+batch_size}: {e}")
    print(f"\n✅ Successfully uploaded {len(documents)} documents to Pinecone.")

# Load vector store for searching
def load_vector_store_sahi_muslim():
    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=pinecone_api
    )

# Build modern QA chain using RunnableMap
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
    prompt = PromptTemplate.from_template(prompt_template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnableMap({
            "context": lambda x: format_docs(x["input_documents"]),
            "question": lambda x: x["question"]
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Handle user query
def user_query_sahi_muslim(query):
    vector_store = load_vector_store_sahi_muslim()
    docs = vector_store.similarity_search(query, k=10)
    chain = get_conversational_chain_sahi_muslim()
    return chain.invoke({"input_documents": docs, "question": query})
