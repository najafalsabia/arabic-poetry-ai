#!/usr/bin/env python
# coding: utf-8

# NOTE: run this in a shell before executing the script, not as part of it:
# pip install -q chromadb sentence-transformers

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

df = pd.read_csv("data/full_final_poems.csv")

def create_document(row):
    return f"""
Title: {row['title']}

Poet: {row['poet_name']}

Era: {row['era']}

Theme: {row['theme']}

Poem:
{row['full_text']}
"""

df["document"] = df.apply(create_document, axis=1)

print(df["document"].iloc[0])

df["full_text"].str.len().describe()

verses_per_poem = df["full_text"].apply(
    lambda x: len([v for v in x.split("\n") if v.strip()])
)

verses_per_poem.describe()

def chunk_poem(poem, max_verses=40, overlap=5):
    verses = [v.strip() for v in poem.split("\n") if v.strip()]

    # Keep normal poems as one document
    if len(verses) <= max_verses:
        return ["\n".join(verses)]

    # Split very long poems
    chunks = []

    start = 0

    while start < len(verses):
        end = start + max_verses

        chunk = "\n".join(verses[start:end])
        chunks.append(chunk)

        if end >= len(verses):
            break

        start += max_verses - overlap

    return chunks

test_chunks = chunk_poem(df.iloc[0]["full_text"])

print(len(test_chunks))
print(test_chunks[0][:500])

model = SentenceTransformer("BAAI/bge-m3")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="arabic_poetry"
)

df_part = df.reset_index(drop=True)

# **Prepare Documents:**

documents = []
metadatas = []
ids = []

for _, row in df_part.iterrows():

    chunks = chunk_poem(row["full_text"])

    for i, chunk in enumerate(chunks):

        document = f"""
Title: {row['title']}

Poet: {row['poet_name']}

Era: {row['era']}

Theme: {row['theme']}

Poem:
{chunk}
"""

        documents.append(document)

        metadatas.append({
            "poet": row["poet_name"],
            "title": row["title"],
            "era": row["era"],
            "theme": row["theme"],
            "source": row["source_url"]
        })

        ids.append(
            f"{row['poem_id']}_{i}"
        )

print(len(documents))

# **Generate Embeddings:**

model = SentenceTransformer("BAAI/bge-m3")

test_embedding = model.encode(documents[0])

print(test_embedding.shape)

import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="arabic_poetry"
)

try:
    client.delete_collection("arabic_poetry")
except:
    pass

collection = client.get_or_create_collection("arabic_poetry")

# Indexing:

batch_size = 64

for i in range(0, len(documents), batch_size):

    batch_docs = documents[i:i + batch_size]
    batch_meta = metadatas[i:i + batch_size]
    batch_ids = ids[i:i + batch_size]

    embeddings = model.encode(
        batch_docs,
        batch_size=16,
        show_progress_bar=False
    ).tolist()

    collection.add(
        ids=batch_ids,
        documents=batch_docs,
        embeddings=embeddings,
        metadatas=batch_meta
    )

    print(f"Indexed {min(i + batch_size, len(documents))}/{len(documents)}")

print(collection.count())
