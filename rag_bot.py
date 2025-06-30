import os
from langchain.embeddings import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from chroma import populate
from parse_web_page import get_chunks_from_webpage


chunks = get_chunks_from_webpage(url="https://fr.wikipedia.org/wiki/Karl_Marx")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = populate(chunks, embeddings)


llm = ChatOpenAI(
    model="llama-3.3-70b-instruct",
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_API_URL"),
)

# 2. Mémoire de conversation
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="answer"
)

# 3. Retriever à partir du vectorstore
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 4. Chaîne RAG conversationnelle
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    output_key="answer",
)


print(
    "🧠 Chatbot RAG prêt et instructionné. Pose ta question pressante (ou tape quit() pour quitter)"
)

while True:
    try:
        query = input("\n❓ Question : ")
        if query.strip().lower() == "quit()":
            print("👋 À bientôt.")
            break
        res = qa_chain.invoke({"question": query})
        print(f"💬 Réponse : {res["answer"]}")
        print(res)

        retrieved_docs = retriever.get_relevant_documents(query)
        print(f"\n🧩 Documents retrouvés pour « {query} » : {len(retrieved_docs)}")
        for d in retrieved_docs:
            print(d.page_content[:300])

    except KeyboardInterrupt:
        print("\n👋 Interruption. À bientôt.")
        break
