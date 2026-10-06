#load pdf
#split into chunks
#create embeddings
#store in vector db


from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import TokenTextSplitter, CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
data=PyPDFLoader("/Users/subhransu/Desktop/untitled folder/rag project/deep_learning_book.pdf")
docs=data.load()




splitter= RecursiveCharacterTextSplitter(
     chunk_size=1000, chunk_overlap=200
)



chunks=splitter.split_documents(docs)


for c in chunks:
    c.page_content = c.page_content.encode('utf-8', 'ignore').decode('utf-8')

embedding_model=GoogleGenerativeAIEmbeddings(model ="gemini-embedding-001")



vectorstore=Chroma.from_documents(
  documents=chunks,
  embedding=embedding_model,
  persist_directory="chroma-db"
)

 