"""
vector_lookup.py — finds which full poem a short user-typed fragment
belongs to, using the shared ChromaDB vector database (BAAI/bge-m3),
then fetches the complete poem text from the CSV by poem_id.
"""
import os
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "..", "vectordb")

# ---- confirmed from build_db.ipynb ----
COLLECTION_NAME = "arabic_poetry"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
DISTANCE_THRESHOLD = 0.35   # starting point — tune once you can test known vs. unknown fragments
# ----------------------------------------

_client = None
_collection = None
_model = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        _collection = _client.get_collection(name=COLLECTION_NAME)
    return _collection


def _get_model():
    """Loads BAAI/bge-m3 once and reuses it (first call downloads the model)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def find_original_poem_vector(verse_fragment: str, df, top_k: int = 1):
    """
    Finds which poem a fragment belongs to via the vector DB, then returns
    the FULL poem text from the CSV using the matched poem_id — this avoids
    returning only a partial chunk for long, split-up poems.

    df : the pandas DataFrame loaded from full_final_poems.csv. Required,
         because the collection's own metadata doesn't include "poet_name"
         or "full_text" — we only use the vector DB to find the poem_id,
         then look up everything else from the CSV, which is guaranteed
         complete and uses consistent column names across your whole team.

    Returns dict {"full_text", "poet_name", "era", "title", "distance"} or None.
    """
    collection = _get_collection()
    model = _get_model()

    query_embedding = model.encode(verse_fragment).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    if not results["ids"] or not results["ids"][0]:
        return None

    distance = results["distances"][0][0]
    if distance > DISTANCE_THRESHOLD:
        return None  # closest match still too far -> treat as "not found"

    matched_id = results["ids"][0][0]   # e.g. "50_0"  ->  poem_id "50", chunk 0
    poem_id = matched_id.split("_")[0]

    row = df[df["poem_id"].astype(str) == poem_id]
    if row.empty:
        return None
    row = row.iloc[0]

    return {
        "full_text": row["full_text"],
        "poet_name": row["poet_name"],
        "era": row["era"],
        "title": row["title"],
        "distance": distance,
    }
