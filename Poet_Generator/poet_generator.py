"""
poet_generator.py - Task 1: Generate a new poem in a chosen poet's style.

Given a poet, theme, and topic, retrieves real example poems by that poet
(filtered by exact poet + theme match in the shared vector database), then
asks an LLM to write a brand new poem that captures the style - vocabulary,
rhythm, imagery - without copying any line from the examples.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# resolve the shared vectordb path relative to this file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_PATH = os.path.join(BASE_DIR, "..", "vectordb")

COLLECTION_NAME = "arabic_poetry"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
GENERATION_MODEL = "gpt-4.1-mini"

_client = None
_collection = None
_model = None
_openai_client = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        _collection = _client.get_collection(COLLECTION_NAME)
    return _collection


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _openai_client


def _build_style_examples(poet: str, theme: str, top_k: int = 3, max_verses: int = 12) -> str:
    """Retrieves top_k real poems by this poet+theme, keeps the first
    max_verses lines of each, and formats them as labeled examples."""
    collection = _get_collection()
    model = _get_model()

    # accept either "وطنية" or "قصائد وطنية" without doubling the prefix
    clean_theme = theme.strip()
    if clean_theme.startswith("قصائد"):
        clean_theme = clean_theme.replace("قصائد", "", 1).strip()

    theme_filter = f"قصائد {clean_theme}"
    query = f"قصائد {poet} في {clean_theme}"
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        where={
            "$and": [
                {"poet": poet},
                {"theme": theme_filter}
            ]
        },
        n_results=top_k
    )

    examples = ""
    for i, doc in enumerate(results["documents"][0], 1):
        poem = doc.split("Poem:")[-1].strip()
        verses = poem.split("\n")
        poem = "\n".join(verses[:max_verses])
        examples += f"\nExample {i}\n\n{poem}\n\n----------------------------------------\n\n"

    return examples


def _build_generation_prompt(poet: str, theme: str, topic: str, examples: str) -> str:
    clean_theme = theme.strip()
    if clean_theme.startswith("قصائد"):
        clean_theme = clean_theme.replace("قصائد", "", 1).strip()
    theme_filter = f"قصائد {clean_theme}"
    return f"""
You are an expert Arabic poet.

Study the STYLE of the following poems.

Do NOT copy any verse.
Do NOT repeat any sentence.
Do NOT continue the examples.

Learn only:
- vocabulary
- rhythm
- imagery
- expressions

Style Examples:

{examples}

===================================

Now write a completely NEW poem.

Topic: {topic}
Poet style: {poet}
Theme: {theme_filter}

Requirements:
- Around 10 verses.
- Classical Arabic.
- Original wording.
- Do not copy any line from the examples.

Return ONLY the poem.
"""


def generate_poem(poet: str, theme: str, topic: str, temperature: float = 0.9) -> dict:
    """
    Generates a new poem in the given poet's style, on the given topic.

    Args:
        poet: exact poet name as stored in the database (e.g. "المتنبي")
        theme: exact theme as stored (e.g. "حكمة", "غزل", "هجاء")
        topic: free-form topic for the new poem (e.g. "الصبر")
        temperature: creativity level for generation

    Returns:
        dict with: poem_text, examples_used, poet, theme, topic
    """
    examples = _build_style_examples(poet, theme)

    if not examples.strip():
        return {
            "poem_text": None,
            "examples_used": 0,
            "poet": poet,
            "theme": theme,
            "topic": topic,
            "error": "لم يتم العثور على قصائد لهذا الشاعر وهذا الثيم في قاعدة البيانات."
        }

    prompt = _build_generation_prompt(poet, theme, topic, examples)
    client = _get_openai_client()

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )

    return {
        "poem_text": response.choices[0].message.content.strip(),
        "examples_used": examples.count("Example"),
        "poet": poet,
        "theme": theme,
        "topic": topic,
        "error": None
    }


if __name__ == "__main__":
    # quick standalone test
    result = generate_poem(poet="المتنبي", theme="حكمة", topic="الصبر")
    if result["error"]:
        print(result["error"])
    else:
        print(f"Poet: {result['poet']} | Theme: {result['theme']} | Topic: {result['topic']}")
        print(f"Examples used: {result['examples_used']}\n")
        print(result["poem_text"])