import os
import shutil
import argparse
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

#--------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Create the vector database for your document")
parser.add_argument("--data", type=str, default="./data/paper1.pdf", help="Path to your document")
parser.add_argument("--chroma_path", type=str, default="./chroma", help="Path to save your vector database")
parser.add_argument("--GOOGLE_API_KEY", type=str, default="YOUR_GEMINI_API_KEY", help="Your Gemini API Key")
args = parser.parse_args()
if args.GOOGLE_API_KEY != "YOUR_GEMINI_API_KEY":
    pass
else:
    raise ValueError("Please enter your Gemini API Key.")

#--------------------------------------------------------------------

#Load pdf file
def load_documents():
    if args.data.split("/")[-1].endswith('pdf'):
        loader = PyPDFLoader(args.data)
        pages = loader.load_and_split()
    elif args.data.split("/")[-1].endswith('txt'):
        loader = TextLoader(args.data, encoding='utf8')
        pages = loader.load()
    else:
        ValueError("Only support for .pdf or .txt, please replace for file.")
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

    if os.path.exists(args.chroma_path):
        shutil.rmtree(args.chroma_path)

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key = args.GOOGLE_API_KEY)

    vectordb = Chroma.from_documents(
        chunks, 
        embedding_model,
        collection_name = "gemini_collection",
        persist_directory = args.chroma_path
    )
    print(f"Saved {len(chunks)} chunks to {args.chroma_path}.")
    return vectordb

# Generate vector database from documents
def generate_embedding_database():
    documents = load_documents()
    chunks = text_split(documents)
    vectordb = initialize_database(chunks)
    return vectordb

vectordb = generate_embedding_database()