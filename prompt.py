

from dotenv import load_dotenv
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundat_filter_retriever import RedundantFilterRetriever

load_dotenv()

chat = ChatOpenAI()

embeddings = OpenAIEmbeddings()


db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)
retriever = RedundantFilterRetriever(
    embbedings=embeddings, chroma=db
)
# retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(
    llm=chat, retriever=retriever, chain_type="stuff")

result = chain.run("What is an interesting fact about the English language?")

print(result)
