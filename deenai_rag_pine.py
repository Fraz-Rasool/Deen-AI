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
CSV_PATH = "merged_quran.csv"
INDEX_NAME = "quran-index"
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
def load_quran_csv():
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    documents = []
    for _, row in df.iterrows():
        content = (
            f"**Surah {row['Surah Name (English)']} ({row['Surah Name (Arabic)']}), "
            f"Surah {row['Surah Number']}, Ayah {int(row['Ayah Number'])}:**\n"
            f"{row['Ayah Translation']}"
        )
        metadata = {
            "surah_english": row['Surah Name (English)'],
            "surah_arabic": row['Surah Name (Arabic)'],
            "surah_number": row["Surah Number"],
            "ayah_number": row["Ayah Number"],
            "revelation_type": row.get("Revelation Type", "Unknown")
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents  # No chunking needed if ayahs are already concise


# Create vector store using Pinecone
def create_vector_store(documents, batch_size=100):
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
def load_vector_store():
    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=pinecone_api
    )

# QA Prompt
def get_conversational_chain():
    prompt_template = """You are DeenAI, an Islamic assistant helping users with authentic responses directly from the Qur'an.

Given the following context (extracted from the Qur'an), answer the user's question **strictly based** on the Ayah translations provided.

In your answer:
- Clearly present the most relevant **Ayah Translation**.
- Mention the **Surah name (both English and Arabic)**, **Surah number**, and **Ayah number** as reference.
- If the context doesn't answer the question, respond with: "I don't know."

CONTEXT: {context}

QUESTION: {question}

Answer:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

# Handle user query
def user_query(query):
    vector_store = load_vector_store()
    docs = vector_store.similarity_search(query, k=10)
    chain = get_conversational_chain()
    result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    return result["output_text"]

# Console Interface
def main():
    print("📖 DeenAI: Console Qur'an QA (Pinecone-based)")
    index = pc.Index(INDEX_NAME)
    if index.describe_index_stats().total_vector_count == 0:
        print("🔄 Pinecone index is empty. Uploading data...")
        docs = load_quran_csv()
        create_vector_store(docs)
        print("✅ Data uploaded to Pinecone.")
    else:
        print("✅ Pinecone index already contains data.")

    print("\n💬 Ask a question from the Qur'an. Type 'exit' to quit.")
    while True:
        query = input("\n🔍 Your question: ")
        if query.strip().lower() in ["exit", "quit"]:
            print("👋 Exiting DeenAI.")
            break
        print("🤖 Thinking...")
        answer = user_query(query)
        print("\n\n🧠 Answer:\n\n", answer)

if __name__ == "__main__":
    main()
