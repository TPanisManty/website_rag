from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader


def get_chunks_from_webpage(
    url: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Loads a webpage, parses its content, splits it into chunks and returns them.
    """
    print(f"[INFO] Loading page: {url}")
    loader = WebBaseLoader(web_paths=(url,))

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(docs)

    return chunks
