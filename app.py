"""
app.py - shared web interface for the Arabic Poetry AI project.

Serves one page with three tools:
- Semantic search over the poetry database
- Rewrite a poem after a modified first verse (same meter/rhyme)
- Generate a new poem in a chosen poet's style

Run from the project root:
    pip install flask
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import sys
from flask import Flask, render_template, request, jsonify

# make the three task folders importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "semantic_search"))
sys.path.append(os.path.join(BASE_DIR, "rewrite_poem"))
sys.path.append(os.path.join(BASE_DIR, "Poet_Generator"))

from semantic_search import search as run_search
from rewrite_poem import find_original_poem, rewrite_poem_validated
from poet_generator import generate_poem

import pandas as pd

CSV_PATH = os.path.join(BASE_DIR, "full_final_poems.csv")
_df = None


def get_df():
    global _df
    if _df is None:
        _df = pd.read_csv(CSV_PATH)
    return _df


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json()
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "الرجاء إدخال نص للبحث."}), 400

    results = run_search(query, top_k=5)
    return jsonify({"results": results})


@app.route("/api/rewrite", methods=["POST"])
def api_rewrite():
    data = request.get_json()
    fragment = (data.get("fragment") or "").strip()
    modified_first_verse = (data.get("modified_first_verse") or "").strip()

    if not fragment or not modified_first_verse:
        return jsonify({"error": "الرجاء إدخال المقطع والبيت المعدل."}), 400

    df = get_df()
    found = find_original_poem(fragment, df=df)

    if found is None:
        return jsonify({"error": "لم يتم العثور على القصيدة الأصلية لهذا المقطع."}), 404

    result = rewrite_poem_validated(
        found["full_text"],
        modified_first_verse,
        found["poet_name"],
        found["era"],
    )

    return jsonify({
        "poet": found["poet_name"],
        "title": found["title"],
        "era": found["era"],
        "generated_text": result["text"],
        "consistent": result["consistent"],
        "attempts": result["attempts"],
    })


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json()
    poet = (data.get("poet") or "").strip()
    theme = (data.get("theme") or "").strip()
    topic = (data.get("topic") or "").strip()

    if not poet or not theme or not topic:
        return jsonify({"error": "الرجاء تعبئة الشاعر والثيم والموضوع."}), 400

    result = generate_poem(poet=poet, theme=theme, topic=topic)

    if result["error"]:
        return jsonify({"error": result["error"]}), 404

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
