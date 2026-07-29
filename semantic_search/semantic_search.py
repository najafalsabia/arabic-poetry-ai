# semantic_search.py
# Task 3 - Semantic Search over Arabic poetry (no LLM required)
# This module exposes a search() function that returns structured results,
# so it can be imported and used directly by app.py (the shared Streamlit interface).

import os
import torch
import chromadb
from sentence_transformers import SentenceTransformer

# resolve the shared vectordb path relative to this file's location,
# so it works no matter which folder this script sits in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "..", "vectordb")

print("Connecting to database...")
client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
collection = client.get_collection("arabic_poetry")
print(f"Connected. Total poems in database: {collection.count()}")

print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-m3")
print("Model loaded successfully.")


def clean_poem_text(text):
    # replace common verse separators (from the scraped source data) with real line breaks
    text = text.replace("***", "\n").replace("  ", " ")
    return text.strip()


def search(query, top_k=5):
    """
    Run a semantic search over the poetry database.

    Args:
        query: the search text (Arabic or English)
        top_k: number of results to return

    Returns:
        A list of dicts, each with: poet, title, era, theme, poem_text, distance
    """
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    structured_results = []

    for i in range(len(results["documents"][0])):
        doc_text = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        structured_results.append({
            "poet": metadata.get("poet", "Unknown"),
            "title": metadata.get("title", "Untitled"),
            "era": metadata.get("era", ""),
            "theme": metadata.get("theme", ""),
            "poem_text": clean_poem_text(doc_text),
            "distance": distance
        })

    return structured_results


if __name__ == "__main__":
    # standalone test / evaluation run
    # evaluation queries required by the task: love, war, horses, desert, night, friendship
    eval_queries = ["الحب", "الحرب", "الخيل", "الصحراء", "الليل", "الصداقة"]

    output_lines = []

    for q in eval_queries:
        results = search(q)
        output_lines.append(f"Query: {q}")
        output_lines.append("=" * 60)

        for i, r in enumerate(results, 1):
            output_lines.append(f"\nResult {i}")
            output_lines.append(f"Poet: {r['poet']} | Title: {r['title']}")
            output_lines.append(f"Era: {r['era']} | Theme: {r['theme']}")
            output_lines.append(f"Distance: {r['distance']:.4f}")
            output_lines.append(r["poem_text"][:300])
            output_lines.append("-" * 60)

        output_lines.append("")

    # save to file for easy viewing in VS Code (terminal doesn't render Arabic correctly)
    with open("search_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print("\nDone. Open search_results.txt in VS Code to view results.")