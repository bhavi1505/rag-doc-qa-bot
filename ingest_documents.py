import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

DATA_FOLDER = "data"
VECTOR_FOLDER = "vector_store"

def load_documents():
    documents = []

    for file in os.listdir(DATA_FOLDER):
        file_path = os.path.join(DATA_FOLDER, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

        elif file.endswith(".txt"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_FOLDER)


if __name__ == "__main__":
    print("📥 Loading documents...")
    docs = load_documents()

    print(f"📄 Total documents loaded: {len(docs)}")

    print("✂️ Splitting into chunks...")
    chunks = split_documents(docs)

    print(f"🧩 Total chunks created: {len(chunks)}")

    print("📦 Creating vector store...")
    create_vector_store(chunks)

    print("✅ Vector store saved successfully!")