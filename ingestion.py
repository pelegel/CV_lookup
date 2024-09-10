from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv

load_dotenv()

def ingest_files(file_path=None):
    # Load the CVs
    if file_path is None:
        file_path = os.path.join(os.getcwd(), "cvs")

    loader = DirectoryLoader(
        path=os.path.join(os.getcwd(), "cvs"),
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    cvs = loader.load()

    ## Split documents
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # texts = text_splitter.split_documents(cvs)

    # Embedding
    embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'], model="text-embedding-3-large")

    PineconeVectorStore.from_documents(cvs, embeddings, index_name=os.environ['INDEX_NAME'])
    print("Ingested all CVs!")


if __name__ == '__main__':
    ingest_files()