import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma

from vector_db import VectorDBGenerator
#-----------------------------------------------------------

class SimpleChatbot(object):
    def __init__(self, vector_db: str, ):
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.vector_db = vector_db
        self.llm = ChatGoogleGenerativeAI(
            model = "gemini-1.5-pro",
            google_api_key= self.GOOGLE_API_KEY,
            temperature= 0.01,
            max_output_tokens=100,
            safety_settings = {
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, 
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, 
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE, 
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            },
        )

    def _create_prompt(self):
        system_prompt = (
            "You are an excellent AI assisstant. Answer the user's question correctly and clearly. "
            "If you don't know the answer, say you don't know. "
            "Use three sentence maximum and keep the answer concise. "
            "No yapping. "
            "If human question is not in context, only answer your knowledge of world truth"
            "Context: {context}"
        )
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{question}"),
            ]
        )

        return prompt

    def _create_chain(self, prompt):
        llm_chain = RetrievalQA.from_chain_type(
            llm = self.llm,
            chain_type= "stuff",
            retriever = self.vector_db.as_retriever(search_type="mmr", 
                                        search_kwargs = {"k":5},
                                        threshold = 0.7,
                                        max_tokens_limit=1024),
            return_source_documents = False,
            chain_type_kwargs= {'prompt': prompt}
        )
        return llm_chain

    def qachatbot(self, question):
        prompt = self._create_prompt()
        llm_chain = self._create_chain(prompt)
        response = llm_chain.invoke({"query": question})
        return response['result']
