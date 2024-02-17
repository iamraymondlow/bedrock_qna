import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.llms import Bedrock


def index():
    # Define data source and load data with PDFLoader
    data_load = PyPDFLoader(
        "https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf"
    )

    # Split the text based on character, tokens, etc recursively
    data_split = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""], chunk_size=100, chunk_overlap=10
    )

    # Create connection to client for vector embeddings
    data_embeddings = BedrockEmbeddings(
        credentials_profile_name="default", model_id="amazon.titan-embed-text-v1"
    )

    # Create vector DB, store embeddings and index for search
    data_index = VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embeddings,
        vectorstore_cls=FAISS,
    )

    # Create index for HR policy document
    db_index = data_index.from_loaders([data_load])

    return db_index


def llm():
    # Connect to Bedrock foundation model - Claude Foundation Model
    llm = Bedrock(
        credentials_profile_name="default",
        model_id="amazon.titan-text-lite-v1",
        model_kwargs={
            "maxTokenCount": 512,
            "temperature": 0.1,
            "topP": 0.9,
        },
    )
    return llm


def rag_response(index, question):
    rag_llm = llm()
    rag_query = index.query(question=question, llm=rag_llm)
    return rag_query
