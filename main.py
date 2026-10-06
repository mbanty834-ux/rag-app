


from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import TokenTextSplitter, CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.vectorstores import Chroma


load_dotenv()

embedding_model=GoogleGenerativeAIEmbeddings(model ="gemini-embedding-001")

vectorstore=Chroma(
    persist_directory="chroma-db",
      embedding_function=embedding_model
      )


retriever=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":3,
          "fetch_k":10,
          "lambda_mult":0.5
          }
          
          
          
   )

llm=ChatGoogleGenerativeAI(model="gemini-3.6-flash")





#prompt template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful assistant. You will be provided with context and a question. Use the context to answer the question. If the context does not contain the answer, say "I don't know" or "The answer is not available in the provided context." Do not make up an answer.
        
        
        
        """),
        ("human",
            """Context: {context}
Question: {question}"""
        )
    ]
)









import re

def clean_text(text):
    # Remove very long base64/encoded strings
    text = re.sub(r'[A-Za-z0-9+/]{100,}={0,2}', '', text)

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


context = "\n\n".join(
    clean_text(doc.page_content)
    for doc in docs
    if doc.page_content
)




print("Rag project is running...")

print("press 0 to exit")


while True:
    query=input("YOU: ")
    if query=="0":
        break
    docs=retriever.invoke(query)
    context = "\n\n".join(
    clean_text(doc.page_content)
    for doc in docs
    if doc.page_content and len(doc.page_content.strip()) > 20
)
    
    final_prompt=prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    response=llm.invoke(final_prompt)
    print(" \n AI: ",response.content)