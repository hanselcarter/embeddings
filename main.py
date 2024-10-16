import chunk
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import argparse
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

load_dotenv()

embeddings = OpenAIEmbeddings()

# emb = embeddings.embed_query("Hello, world!")

# print(emb)

tex_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(tex_splitter)

db = Chroma.from_documents(docs, embedding=embeddings,
                           persist_directory="emb")

results = db.similarity_search_with_score(
    "What is interstic fact about the engligh language?")

for result in results:
    print("\n")
    print(result[1])
    print(result[0].page_content)
