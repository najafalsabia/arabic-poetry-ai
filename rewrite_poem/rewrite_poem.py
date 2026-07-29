"""
rewrite_poem.py — Task 2: Rewrite Poem (Style-Preserving Continuation)

Full flow now:
  1. User types a FRAGMENT they remember (not the whole poem)
  2. find_original_poem() looks it up FAST via your teammate's ChromaDB
     vector DB (vector_lookup.py) -> falls back to a plain text scan over
     the CSV only if the vector DB isn't reachable yet
  3. rewrite_poem() takes the found poem + the user's MODIFIED first verse
     -> builds a prompt with the full original text embedded directly in
     it -> sends to GPT -> returns a new continuation

Note on step 3: this part still does NOT use retrieval/similarity search.
The vector DB in step 2 is only a fast way to fetch text the user already
quoted verbatim — GPT never searches for "similar" poems to write from,
it just reads the one poem you hand it directly in the prompt.

Setup:
    pip install openai pandas chromadb
    Set OPENAI_API_KEY as an environment variable
    (in Colab: from google.colab import userdata; then
     os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY"))
"""

import os
import re
import difflib
import pandas as pd
from openai import OpenAI
from prompts import build_rewrite_prompt
from dotenv import load_dotenv

load_dotenv()


def normalize_arabic(text: str) -> str:
    """Strips diacritics + unifies letter variants so matching isn't broken by
    tashkeel, alef/hamza forms, or ya/alef-maqsura differences."""
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)  # remove diacritics (tashkeel)
    text = text.replace("إ", "ا").replace("أ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_original_poem_textmatch(df: pd.DataFrame, verse_fragment: str):
    """
    FALLBACK ONLY — plain text lookup (normalize + substring/fuzzy match)
    over the raw CSV. Works, but scans every row, so it's noticeably slower
    than the vector lookup once the DB is large. Use find_original_poem()
    below, which prefers the fast vector-DB path and only falls back to
    this if the vector DB isn't reachable yet.

    Returns the matching row (pandas Series) or None if nothing close enough is found.
    """
    frag_norm = normalize_arabic(verse_fragment)

    # 1) fast path: does any poem literally contain this text?
    mask = df["full_text"].apply(lambda t: frag_norm in normalize_arabic(t))
    matches = df[mask]
    if len(matches) >= 1:
        return matches.iloc[0]  # if >1 (rare, shared line), just take the first

    # 2) fallback: fuzzy match against each poem's first line (handles typos/spacing)
    best_score, best_row = 0.0, None
    for _, row in df.iterrows():
        first_line = normalize_arabic(row["full_text"].split("\n")[0])
        score = difflib.SequenceMatcher(None, first_line, frag_norm).ratio()
        if score > best_score:
            best_score, best_row = score, row

    return best_row if best_score >= 0.6 else None


def find_original_poem(verse_fragment: str, df: pd.DataFrame = None):
    """
    Main entry point: finds the original poem a user's fragment came from.

    Tries the FAST vector-DB lookup first (built by your teammate — see
    vector_lookup.py for the confirmed setup: BAAI/bge-m3, collection
    "arabic_poetry"). If that's not reachable (not built yet, wrong path,
    etc.), falls back to the slower text-match search over `df` so you're
    never blocked.

    `df` is required either way now — the vector path uses it to fetch the
    guaranteed-complete full_text by poem_id (see vector_lookup.py's
    docstring for why), and the fallback path scans it directly.

    Returns a dict: {"full_text", "poet_name", "era", "title"} or None.
    """
    if df is None:
        raise ValueError("`df` (the poems CSV) is required — pass it in either way.")

    try:
        from vector_lookup import find_original_poem_vector
        result = find_original_poem_vector(verse_fragment, df=df)
        if result is not None:
            return result
    except Exception as e:
        print(f"(vector DB lookup unavailable, falling back to text search — {e})")

    row = find_original_poem_textmatch(df, verse_fragment)
    if row is None:
        return None
    return {
        "full_text": row["full_text"],
        "poet_name": row["poet_name"],
        "era": row["era"],
        "title": row["title"],
    }

MODEL = "gpt-4o"  # classical Arabic poetry needs the stronger model — mini's meter/rhyme quality wasn't reliable enough.


def rewrite_poem(original_poem: str, modified_first_verse: str,
                  poet_name: str = None, era: str = None,
                  model: str = MODEL, temperature: float = 0.9) -> str:
    """
    Calls the LLM and returns only the new continuation text (str).
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    messages = build_rewrite_prompt(original_poem, modified_first_verse, poet_name, era)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=400,
    )

    return response.choices[0].message.content.strip()


def rewrite_poem_validated(original_poem: str, modified_first_verse: str,
                            poet_name: str = None, era: str = None,
                            model: str = MODEL, temperature: float = 0.9,
                            max_attempts: int = 3) -> dict:
    """
    Same as rewrite_poem(), but checks the rhyme (validate_rhyme) and
    automatically retries with a corrective note if it drifts — instead of
    just hoping the prompt instruction was enough.

    Returns {"text": str, "consistent": bool, "attempts": int, "mismatches": list}
    so you can log/report exactly how often this happens for the eval report.
    """
    from prompts import extract_rawi, validate_rhyme

    rawi = extract_rawi(modified_first_verse)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    messages = build_rewrite_prompt(original_poem, modified_first_verse, poet_name, era)

    for attempt in range(1, max_attempts + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=400,
        )
        text = response.choices[0].message.content.strip()
        check = validate_rhyme(text, rawi)

        if check["consistent"] or attempt == max_attempts:
            return {
                "text": text,
                "consistent": check["consistent"],
                "attempts": attempt,
                "mismatches": check["mismatches"],
            }

        # feed the exact mismatches back so the retry targets the real problem
        correction_note = (
            f"المحاولة السابقة انحرفت عن حرف الروي \"{rawi}\" بهذي الأبيات:\n"
            + "\n".join(check["mismatches"])
            + f"\n\nأعد المحاولة والتزم بحرف الروي \"{rawi}\" في نهاية كل شطر ثانٍ بدون استثناء."
        )
        messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": correction_note},
        ]

    # unreachable, but keeps linters happy
    return {"text": text, "consistent": False, "attempts": max_attempts, "mismatches": check["mismatches"]}


if __name__ == "__main__":
    # ---- Full realistic flow: user types a fragment they know, not the whole poem ----
    # `df` here is only used as the FALLBACK if the vector DB isn't set up yet.
    # Once you plug in your teammate's real vectordb/ (see vector_lookup.py), this
    # same call automatically uses the fast path — no code change needed here.
    df = pd.read_csv("full_final_poems.csv")

    # this is what the USER actually types into the app — a fragment they remember
    user_typed_fragment = "قفا نبك من ذكرى حبيب"

    found = find_original_poem(user_typed_fragment, df=df)
    if found is None:
        print("ما لقيت قصيدة تطابق هالمقطع — جربي تتأكدين من الإملاء أو التشكيل.")
    else:
        original_poem = found["full_text"]
        poet_name = found["poet_name"]
        era = found["era"]
        print(f"لقيت القصيدة: {found['title']} — {poet_name} ({era})")

        # the user's NEW first verse (their creative edit) — separate from the lookup fragment
        modified_first_verse = "قفا نبك من ذكرى حبيب ومنزل *** ولكن على أطلال قلب مكبل"

        result = rewrite_poem(original_poem, modified_first_verse, poet_name, era)
        print("\n=== النتيجة ===")
        print(result)
