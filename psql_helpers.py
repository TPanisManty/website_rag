from langchain.vectorstores.pgvector import PGVector
from urllib.parse import quote_plus
import os

user = os.getenv("DB_POSTGRES_USER")
raw_password = os.getenv("DB_POSTGRES_PWD")
ip = os.getenv("DB_POSTGRES_IP")
port = os.getenv("DB_POSTGRES_PORT")

encoded_password = quote_plus(raw_password)

CONNECTION_STRING = (
    f"postgresql+psycopg2://{user}:{encoded_password}@{ip}:{port}/website_rag"
)


def populate_distant_db(chunks, embeddings):
    return PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="website_rag",
        connection_string=CONNECTION_STRING,
    )
