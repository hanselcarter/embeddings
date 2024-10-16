from typing import Any, Dict, List
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever
from langchain_core.documents import Document


class RedundantFilterRetriever(BaseRetriever):
    embbedings: Embeddings
    chroma: Chroma

    def get_relevant_documents(self, query):
        # calcula embedding for the query string
        emb = self.embbedings.embed_query(query)
        # take embeddings and feeed them into that
        # max_margina_Relavance_Search_by_vector
        return self.chroma.max_marginal_relevance_search_by_vector(embedding=emb, lambda_mult=0.8)

    async def aget_relevant_documents(self):
        return []
