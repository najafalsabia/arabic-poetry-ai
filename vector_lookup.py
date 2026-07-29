"""
vector_lookup.py — Task 2 helper: fast poem lookup using the shared ChromaDB
vector database (built by your team for Task 1/3).

CONFIRMED FROM build_db.ipynb (your teammate's actual notebook):
- Embedding model: BAAI/bge-m3 via sentence-transformers (1024-dim)
- Embeddings were computed manually with model.encode() and passed straight
  to collection.add(embeddings=...) — the collection has NO embedding_function
  attached. That means queries must ALSO manually encode with the same model
  and call collection.query(query_embeddings=[...]), not query_texts=[...].
  (This is exactly why the first attempt failed with "expecting 1024, got 384"
  — Chroma silently used its own default embedder for query_texts.)
- metadata keys stored: "poet", "title", "era", "theme", "source"
  (note: "poet", NOT "poet_name" — and there is NO "full_text" key at all)
- ids are formatted as "{poem_id}_{chunk_index}" — poems longer than 40
  verses get split into overlapping chunks, so one poem can have several ids.
- ⚠️ the indexing loop in the notebook ran on df_test = df.sample(1000,
  random_state=42) — a 1000-POEM TEST SAMPLE, not the full ~128k dataset.
  Confirm with her whether she re-ran it on the full CSV before the final
  submission. If not, most fragments simply won't be in there yet, and
  find_original_poem() in rewrite_poem.py will silently fall back to the
  text-match search over the CSV — so you won't be blocked, just slower.

WHY WE STILL GO BACK TO THE CSV FOR THE FULL TEXT
--------------------------------------------------
Long poems are chunked, so a single vector match might only be ONE piece
of a longer poem, not the whole thing. Instead of parsing partial text out
of the document string, we use the vector match purely to find WHICH
poem_id the fragment belongs to (parsed from the id "{poem_id}_{i}"), then
fetch the guaranteed-complete full_text for that poem_id from the CSV.
"""

import chromadb
from sentence_transformers import SentenceTransformer

# ---- confirmed from build_db.ipynb ----
VECTOR_DB_PATH = "."
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
