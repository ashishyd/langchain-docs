import os
from typing import Any, List, Dict

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docSearch = PineconeVectorStore.from_existing_index(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=docSearch.as_retriever(),
        return_source_documents=True,
    )
    return qa.invoke({"question": query, "chat_history": chat_history})
