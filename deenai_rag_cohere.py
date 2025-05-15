import pandas as pd
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
gem_api = os.getenv("GEMINI_API_KEY")

# Constants
VECTOR_DB_PATH = "faiss_quran_index"
CSV_PATH = "merged_quran.csv"  # your pre-cleaned CSV

# Set up LLM and embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5, google_api_key=gem_api)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gem_api)

# Load and chunk Qur’an CSV
def load_quran_csv():
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    documents = []
    for _, row in df.iterrows():
        content = (
            f"Surah {row['Surah Name (English)']}"
            f"Ayah {int(row['Ayah Number'])}:\n"
            f"{row['Ayah Translation']}"
        )
        documents.append(Document(page_content=content))
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

# Create FAISS vector store and save
def create_vector_store(docs):
    vector_store = FAISS.from_documents(docs, embedding=embeddings)
    vector_store.save_local(VECTOR_DB_PATH)

# Load vector store
def load_vector_store():
    return FAISS.load_local(VECTOR_DB_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)

# Get QA Chain
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

# Console version of main
def main():
    print("📖 DeenAI: Console Qur'an QA")
    if not os.path.exists(f"{VECTOR_DB_PATH}/index.faiss"):
        print("🔄 Initializing vector store from Qur’an CSV...")
        docs = load_quran_csv()
        create_vector_store(docs)
        print("✅ Vector store created and saved.")
    else:
        print("✅ Vector store already exists. Loaded from disk.")

    print("\n💬 You can now ask questions from the Qur'an. Type 'exit' to quit.")
    while True:
        query = input("\n🔍 Your question: ")
        if query.strip().lower() in ["exit", "quit"]:
            print("👋 Exiting DeenAI.")
            break
        print("🤖 Thinking...")
        answer = user_query(query)
        print("\n🧠 Answer:\n", answer)

if __name__ == "__main__":
    main()
