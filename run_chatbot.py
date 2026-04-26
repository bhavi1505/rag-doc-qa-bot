import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
# Load environment variables
load_dotenv()

VECTOR_FOLDER = "vector_store"

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.load_local(
    VECTOR_FOLDER,
    embeddings,
    allow_dangerous_deserialization=True
)

def get_answer(db, query):
    docs = db.similarity_search(query, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the context below.
If the answer is not present, say:
"Answer not found in the documents."

Context:
{context}

Question:
{query}
"""

    # Simple answer from retrieved context
    if context.strip() == "":
        answer = "Answer not found in the documents."
    else:
        answer = context[:500]
      # simple answer from documents

    return answer, docs


if __name__ == "__main__":
    print("🤖 Document Q&A Bot Started! (type 'exit' to quit)")

    db = load_vector_store()

    while True:
        user_query = input("\n❓ Ask your question: ")

        if user_query.lower() == "exit":
            break

        answer, sources = get_answer(db, user_query)

        print("\n💡 Answer:\n", answer)

        print("\n📌 Sources:")
        for i, doc in enumerate(sources):
            print(f"{i+1}. {doc.metadata}")