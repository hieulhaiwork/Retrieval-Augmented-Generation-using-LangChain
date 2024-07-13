import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

DATA_PATH = r"data/paper1.pdf"
CHROMA_PATH = r"chroma"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

#--------------------------------------------------------------------

#Load pdf file
def load_documents():
    loader = PyPDFLoader(DATA_PATH)
    pages = loader.load_and_split()
    return pages

#Split chunks
def text_split(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 500,
        length_function = len,
        add_start_index = True,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks

# Initialize Chroma database
def initialize_database(chunks):

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key = GOOGLE_API_KEY)

    vectordb = Chroma.from_documents(
        chunks, 
        embedding_model,
        collection_name = "gemini_collection",
        persist_directory = CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    return vectordb

# Generate vector database from documents
def generate_embedding_database():
    documents = load_documents()
    chunks = text_split(documents)
    vectordb = initialize_database(chunks)
    return vectordb

vectordb = generate_embedding_database()