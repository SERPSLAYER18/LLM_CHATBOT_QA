import chromadb
from langchain.vectorstores import Chroma

import config
from src.nlp_tools import tokenize


def postprocess(docs):
    texts = [d.page_content for d in docs]
    return texts


def remove_duplicates(texts):
    texts = list(set(texts))
    return texts


def append_source(docs):
    for d in docs:
        source = d.metadata["canonicalUrl"]
        d.page_content = f"Источник: {source} \n{d.page_content}"
    return docs


def top_k_by_tf_ranking(strings, query, top_k=3):
    query_tokens = tokenize(query)
    results = []
    for s in strings:
        string_tokens = tokenize(s)
        tf = 0
        for string_token in string_tokens:
            if string_token in query_tokens:
                tf += 1

        results.append((s, tf))
    results.sort(key=lambda x: x[1], reverse=True)
    top_texts = [r[0] for r in results[:top_k]]
    return top_texts


class ChromaRetriever:

    def __init__(self, embeddings):
        settings = chromadb.config.Settings(
            is_persistent=True, persist_directory=config.CHROMA_DIR
        )
        self.text_db = Chroma(
            collection_name=config.CHROMA_COLLECTION_NAME, client_settings=settings, embedding_function=embeddings
        )
        self.top_k = config.MMR_TOP_K
        self.tf_top_k = config.TF_TOP_K
        self.similarity_top_k = config.SIMILARITY_TOP_K
        self.mmr_lambda = config.MMR_LAMBDA

    def retrieve_docs(self, query):
        retrieved_docs = self.text_db.max_marginal_relevance_search(
            query, fetch_k=self.top_k, k=self.top_k, lambda_mult=self.mmr_lambda
        )
        retrieved_docs = append_source(retrieved_docs)
        retrieved_texts = postprocess(retrieved_docs)
        retrieved_texts = remove_duplicates(retrieved_texts)

        similarity_docs = retrieved_texts[:self.similarity_top_k]
        tf_top_docs = top_k_by_tf_ranking(retrieved_texts, query, self.tf_top_k)
        top_docs = remove_duplicates(tf_top_docs + similarity_docs)

        concated = "\n".join(top_docs)

        return concated
