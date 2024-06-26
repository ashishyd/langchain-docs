import os
from dotenv import load_dotenv
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def ingest_docs() -> None:
    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest")
    raw_documents = loader.load()
    print(f"loader {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", "", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Split into {len(documents)} chunks")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(
        documents=documents, embedding=embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    print("Ingesting...")
    ingest_docs()
