import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="📚 RAG Chatbot", layout="wide")

st.title("📚 Document Chatbot")
st.write("Ask multiple questions based on your documents")

VECTOR_FOLDER = "vector_store"

@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(
        VECTOR_FOLDER,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db

db = load_db()

# 🧠 Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
user_input = st.chat_input("Ask your question...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve answer
    docs = db.similarity_search(user_input, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    if context.strip() == "":
        answer = "Answer not found in the documents."
    else:
        answer = context[:500]

    # Show assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Show sources
    st.markdown("### 📌 Sources")
    sources = set([doc.metadata['source'] for doc in docs])
    for src in sources:
        st.write(f"- {src}")