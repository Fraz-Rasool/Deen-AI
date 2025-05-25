import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
pinecone_api = os.getenv("PINECONE_API_KEY")

# Set up LLM and embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, max_tokens=3000)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# Define index names
indexes = {
    "quran": "quran-index",
    "bukhari": "sahibukhari-index",
    "muslim": "sahimuslim-index"
}

# Helper function to load vector store
def load_vector_store(index_name):
    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=pinecone_api
    )

# Retrieve top documents from each source
def retrieve_docs(query, k=5):
    all_docs = []
    for name, index in indexes.items():
        vs = load_vector_store(index)
        docs = vs.similarity_search(query, k=k)
        all_docs.extend(docs)
    return all_docs

# Create QA summarization chain
def get_summary_chain():
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template = """
You are DeenAI, an Islamic assistant designed to help users by summarizing authentic knowledge derived strictly from the following sources: the Qur'an, Sahih Bukhari, and Sahih Muslim.

Instructions:
- Use **only the given context** to generate your response.
- Summarize the most relevant information in a clear and concise manner.
- **Do not include** direct references to Surah names, Hadith numbers, book titles, or narrators.
- Your response should be a **summary only**, with no citations or religious source formatting.
- Limit the response to a **maximum of 1000 words**.
- Response must give 500 to 600 words.
- If the context does not contain enough information to answer the question, respond with: **"I don't know."**

CONTEXT:
{context}

QUESTION:
{question}

Answer:
"""
    )
    return create_stuff_documents_chain(llm=llm, prompt=prompt)

# Unified interface
def unified_query(question: str) -> str:
    docs = retrieve_docs(question)
    chain = get_summary_chain()
    result = chain.invoke({"context": docs, "question": question})
    return result

# Example usage
if __name__ == "__main__":
    while True:
        q = input("\n🔍 Ask a question (or type 'exit'): ")
        if q.lower().strip() in ["exit", "quit"]:
            break
        print("\n🤖 Generating answer...\n")
        print(unified_query(q))
