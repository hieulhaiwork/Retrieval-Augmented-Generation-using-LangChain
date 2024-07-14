import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from vector_database import vectordb



# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

CHROMA_PATH = r"chroma"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def initialize_llm():
    llm = ChatGoogleGenerativeAI(
        model = "gemini-1.5-pro",
        google_api_key= GOOGLE_API_KEY,
        temperature= 0.01,
        max_output_tokens=100,
        safety_settings = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, 
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, 
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, 
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        },
    )
    return llm

def create_prompt():
    system_prompt = (
        "You are an excellent AI assisstant. Answer the user's question correctly and clearly. "
        "If you don't know the answer, say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "No yapping. "
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )

    return prompt

def read_vector_db():
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key = GOOGLE_API_KEY)
    vectordb = Chroma(persist_directory=CHROMA_PATH, 
                        embedding_function=embeddings_model,
                        collection_name="gemini_collection")
    #db = VectorStoreRetriever(vectorstore=vectordb)
    return vectordb

def create_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = db.as_retriever(search_type="mmr", 
                                    search_kwargs = {"k":5},
                                    threshold = 0.7,
                                    max_tokens_limit=1024),
        return_source_documents = False,
        chain_type_kwargs= {'prompt': prompt}

    )
    return llm_chain

def qachatbot(question):
    db = read_vector_db()
    llm = initialize_llm()
    prompt = create_prompt()

    llm_chain = create_chain(prompt, llm, db)

    question = "Who is the author of this paper?"
    response = llm_chain.invoke({"query": question})
    print(response['result'])