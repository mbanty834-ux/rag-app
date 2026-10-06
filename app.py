```python
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🤖 AI RAG Assistant")
st.write("Ask questions from your knowledge base.")


# --------------------------------------------------
# Load Embedding Model
# --------------------------------------------------

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# --------------------------------------------------
# Load Chroma Vector Database
# --------------------------------------------------

vectorstore = Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model
)


# --------------------------------------------------
# Create Retriever
# --------------------------------------------------

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)


# --------------------------------------------------
# Load Gemini LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.2
)


# --------------------------------------------------
# RAG Prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful AI RAG assistant.

            Answer the user's question using ONLY the provided context.

            If the answer is not present in the context, say:

            "I don't know. The answer is not available
            in the provided context."

            Do not make up information.

            Context:
            {context}
            """
        ),
        (
            "human",
            "{question}"
        )
    ]
)


# --------------------------------------------------
# Chat History
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Display Previous Messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# User Input
# --------------------------------------------------

query = st.chat_input("Ask something about your documents...")


if query:

    # Display user question
    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )


    # --------------------------------------------------
    # Retrieve relevant documents
    # --------------------------------------------------

    with st.spinner("Searching knowledge base..."):

        docs = retriever.invoke(query)


    # --------------------------------------------------
    # Create Context
    # --------------------------------------------------

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )


    # --------------------------------------------------
    # Create Final Prompt
    # --------------------------------------------------

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )


    # --------------------------------------------------
    # Generate Answer
    # --------------------------------------------------

    with st.spinner("Generating answer..."):

        response = llm.invoke(final_prompt)

        answer = response.content


    # --------------------------------------------------
    # Display AI Response
    # --------------------------------------------------

    with st.chat_message("assistant"):

        st.markdown(answer)


    # --------------------------------------------------
    # Save AI Response
    # --------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


    # --------------------------------------------------
    # Show Retrieved Sources
    # --------------------------------------------------

    with st.expander("📚 Retrieved Documents"):

        for i, doc in enumerate(docs, start=1):

            st.markdown(f"### Document {i}")

            st.write(doc.page_content)

            if doc.metadata:
                st.caption(
                    f"Metadata: {doc.metadata}"
                )


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ RAG Settings")

    st.write("**Embedding Model**")
    st.code("gemini-embedding-001")

    st.write("**LLM**")
    st.code("gemini-3.6-flash")

    st.write("**Retriever**")
    st.code("MMR")

    st.write("**Documents Retrieved**")
    st.write("3")

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()
```
