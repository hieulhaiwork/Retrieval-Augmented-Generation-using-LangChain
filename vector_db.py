import os
import shutil

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

#-------------------------------------------------------------------

class VectorDBGenerator(object):
    def __init__(self, db_path: str, data_path: str, embedding_model: str):
        self.db_path = db_path
        self.data_path = data_path
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.embedding_model = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key = self.GOOGLE_API_KEY)

    def _load_documents(self):
        if self.data_path.endswith('pdf'):
            loader = PyPDFLoader(self.data_path)
            pages = loader.load_and_split()
        elif self.data_path.endswith('txt'):
            loader = TextLoader(self.data_path, encoding='utf8')
            pages = loader.load()
        else:
            raise ValueError(f"Only support for .pdf or .txt.")
        return pages

    def _text_split(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 500,
            length_function = len,
            add_start_index = True,
        )

        chunks = text_splitter.split_documents(documents)
        # print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    def _initialize_database(self, chunks, db_name):

        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)

        vectordb = Chroma.from_documents(
            chunks, 
            self.embedding_model,
            # collection_name = collection_name,
            persist_directory = os.path.join(self.db_path, db_name) 
        )
        # print(f"Saved {len(chunks)} chunks to {self.db_path}.")
        return vectordb

    def generate_vectordb(self, db_name: str):
        documents = self._load_documents()
        chunks = self._text_split(documents)
        vectordb = self._initialize_database(chunks, db_name)
        return vectordb

    def load_vectordb(self):
        vectordb = Chroma(persist_directory=self.data_path,
                        embedding_function=self.embedding_model,
                        )
        return vectordb