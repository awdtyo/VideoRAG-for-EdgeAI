import chromadb
from sentence_transformers import SentenceTransformer

import config


class Retriever:
    def __init__(self):
        self.text_embedder = SentenceTransformer(config.TEXT_EMBED_MODEL, device="cpu")
        client = chromadb.PersistentClient(path=config.DB_DIR)
        self.collection = client.get_or_create_collection(config.CHROMA_COLLECTION)

    def query(self, question, top_k=config.TOP_K_RETRIEVE, video_id=None):
        q_vec = self.text_embedder.encode([question], normalize_embeddings=True).tolist()

        where = {"video_id": video_id} if video_id else None
        results = self.collection.query(
            query_embeddings=q_vec, n_results=top_k, where=where
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            hits.append({"text": doc, "meta": meta, "score": 1 - dist})
        return hits
