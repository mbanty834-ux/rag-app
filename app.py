import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        color: #777;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .status-box {
        padding: 12px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 RAG Assistant")

    st.markdown("---")

    st.subheader("📚 Knowledge Base")

    st.success("Vector Database Connected")

    st.caption("Database: ChromaDB")
    st.caption("Embedding: Gemini Embedding")
    st.caption("Retriever: MMR")

    st.markdown("---")

    st.subheader("⚙️ RAG Settings")

    k_value = st.slider(
        "Documents to retrieve",
        min_value=1,
        max_value=10,
        value=3
    )

    fetch_k_value = st.slider(
        "Candidate documents",
        min_value=5,
        max_value=30,
        value=10
    )

    lambda_value = st.slider(
        "MMR Diversity",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1
    )

    st.markdown("---")

    st.subheader("🤖 AI Model")

    st.info("Gemini Flash")

    st.caption("Temperature: 0.7")

    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()

    st.markdown("---")

    st.caption("RAG Knowledge Assistant")
    st.caption("Powered by LangChain + Gemini")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧠 RAG Knowledge Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask questions from your documents using Retrieval-Augmented Generation</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

@st.cache_resource
def load_embedding_model():

    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )


embedding_model = load_embedding_model()


# ============================================================
# LOAD VECTOR DATABASE
# ============================================================

@st.cache_resource
def load_vectorstore():

    return Chroma(
        persist_directory="chroma-db",
        embedding_function=embedding_model
    )


vectorstore = load_vectorstore()


# ============================================================
# RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": k_value,
        "fetch_k": fetch_k_value,
        "lambda_mult": lambda_value
    }
)


# ============================================================
# LOAD LLM
# ============================================================

@st.cache_resource
def load_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0.7
    )


llm = load_llm()


# ============================================================
# PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful RAG assistant.

Answer the user's question using ONLY the provided context.

If the answer is not available in the context, say:

"I don't know. The answer is not available in the provided context."

Do not hallucinate or make up information.

Context:
{context}
"""
        ),
        (
            "human",
            "Question: {question}"
        )
    ]
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# WELCOME SCREEN
# ============================================================

if len(st.session_state.messages) == 0:

    st.info(
        "👋 Welcome! Ask a question about the documents stored in your knowledge base."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📖 Ask Questions")
        st.caption("Ask questions about your documents.")

    with col2:
        st.markdown("### 🔎 Semantic Search")
        st.caption("Find relevant information using embeddings.")

    with col3:
        st.markdown("### 🤖 AI Answers")
        st.caption("Gemini generates answers from retrieved context.")


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:

            with st.expander("🔎 View Retrieved Context"):

                for i, source in enumerate(
                    message["sources"],
                    start=1
                ):

                    st.markdown(
                        f"**Document {i}**"
                    )

                    st.write(source)


# ============================================================
# USER INPUT
# ============================================================

query = st.chat_input(
    "💬 Ask something about your documents..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):

        st.markdown(query)


    # --------------------------------------------------------
    # RETRIEVE DOCUMENTS
    # --------------------------------------------------------

    with st.spinner("🔎 Searching your knowledge base..."):

        docs = retriever.invoke(query)


    # --------------------------------------------------------
    # CREATE CONTEXT
    # --------------------------------------------------------

    context = "\n\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )


    # --------------------------------------------------------
    # CREATE PROMPT
    # --------------------------------------------------------

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )


    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("🤖 Generating answer..."):

            response = llm.invoke(final_prompt)

            answer = response.content

        st.markdown(answer)

        # ----------------------------------------------------
        # RETRIEVED SOURCES
        # ----------------------------------------------------

        with st.expander("🔎 View Retrieved Context"):

            if docs:

                for i, doc in enumerate(
                    docs,
                    start=1
                ):

                    st.markdown(
                        f"### 📄 Document {i}"
                    )

                    st.write(
                        doc.page_content
                    )

                    if doc.metadata:

                        st.caption(
                            f"Metadata: {doc.metadata}"
                        )

            else:

                st.warning(
                    "No relevant documents were found."
                )


    # ========================================================
    # SAVE MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": [
                doc.page_content
                for doc in docs
            ]
        }
    )