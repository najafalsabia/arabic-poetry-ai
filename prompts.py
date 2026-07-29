"""
prompts.py — Task 2: Rewrite Poem (Style-Preserving Continuation)

Builds the LLM prompt used to continue a poem after the user changes its
first verse. The plan notes that few-shot prompting works noticeably
better than a plain instruction, so this module ships one worked example
and appends the user's real case after it.

UPDATE: added extract_rawi() after testing showed GPT drifting off the
rhyme letter (qafiya) after the first couple of verses — e.g. ending on
ل then drifting to ي and ه. Telling the model the exact rawi letter
explicitly (instead of hoping it infers and holds it) fixes this.
"""

import re


def extract_rawi(verse: str) -> str:
    """
    Extracts the rhyme letter (رَوِي) from a verse — the last consonant of
    the second hemistich, which every following verse must end on.

    e.g. "... ولكن على أطلال قلب مكبل" -> "ل"
    """
    verse = re.sub(r"[\u064B-\u065F\u0670]", "", verse)  # strip diacritics
    second_hemistich = verse.split("***")[-1].strip()
    return second_hemistich[-1] if second_hemistich else ""

SYSTEM_PROMPT = """أنت خبير في الشعر العربي الكلاسيكي، متمكن من العروض والبلاغة وأساليب الشعراء عبر العصور.
مهمتك: يقوم المستخدم بتغيير البيت الأول من قصيدة معروفة، وعليك أن تكتب استمرارًا جديدًا للقصيدة
ينسجم مع هذا البيت الجديد.

القواعد الصارمة:
- حافظ على الأسلوب الشعري والوزن والقافية المناسبة للسياق الكلاسيكي.
- لا تنسخ أي بيت من القصيدة الأصلية حرفيًا.
- لا تكرر نفس المعنى أو الصور بشكل مباشر من الأصل.
- اجعل الأبيات الجديدة متماسكة المعنى ومترابطة مع البيت الأول المعدل.
- أعد فقط الأبيات الجديدة الجديدة بدون أي شرح أو مقدمات."""

FEW_SHOT_EXAMPLES = [
    {
        "original_first_verse": "أنا الذي تعرف البطحاء وطأته *** والبيت يعرفه والحل والحرم",
        "modified_first_verse": "أنا الذي تعرف العمياء نبرته *** والصمت يدركه واللفظ والكلم",
        "continuation": (
            "أمشي فتنحني الدروب مهابة *** وتصمت الأصوات حين أقم\n"
            "لا الجهل يعرفني ولا يخفى صدى *** صوتي إذا ما لجّ فيه الظلم\n"
            "أنا صوت من لم تبصر الدنيا له *** نورًا، ولكن سمعها لا ينعدم"
        ),
    },
]


def validate_rhyme(generated_text: str, expected_rawi: str) -> dict:
    """
    Checks whether every verse in the generated continuation actually ends
    with the expected rawi letter. Useful both for a quick sanity check
    while developing, and as a hard number for the evaluation report
    (Weekly Project requires reporting where the system fails and why).

    Returns {"consistent": bool, "line_endings": [...], "mismatches": [...]}.
    """
    lines = [l.strip() for l in generated_text.split("\n") if l.strip()]
    endings = []
    mismatches = []

    for line in lines:
        clean = re.sub(r"[\u064B-\u065F\u0670]", "", line)
        second_hemistich = clean.split("***")[-1].strip()
        ending = second_hemistich[-1] if second_hemistich else ""
        endings.append(ending)
        if ending != expected_rawi:
            mismatches.append(line)

    return {
        "consistent": len(mismatches) == 0,
        "line_endings": endings,
        "mismatches": mismatches,
    }
def build_rewrite_prompt(original_poem: str, modified_first_verse: str,
                          poet_name: str = None, era: str = None) -> list:
    """
    Builds the chat-format `messages` list to send to the LLM.

    original_poem          : full_text of the original poem
                              (from the CSV now, or from your teammate's
                              vector DB metadata later — same string either way)
    modified_first_verse   : the new first verse the user typed in the UI
    poet_name / era        : optional, helps the model imitate style more precisely
    """
    context_line = ""
    if poet_name:
        context_line += f"الشاعر: {poet_name}\n"
    if era:
        context_line += f"العصر: {era}\n"

    rawi = extract_rawi(modified_first_verse)

    few_shot_text = ""
    for ex in FEW_SHOT_EXAMPLES:
        few_shot_text += (
            f"مثال:\nالبيت الأول الأصلي: {ex['original_first_verse']}\n"
            f"البيت الأول المعدل: {ex['modified_first_verse']}\n"
            f"الاستمرار المطلوب:\n{ex['continuation']}\n\n"
        )

    user_prompt = (
        f"{context_line}"
        f"القصيدة الأصلية كاملة:\n{original_poem}\n\n"
        f"{few_shot_text}"
        f"الآن، المستخدم غيّر البيت الأول إلى:\n{modified_first_verse}\n\n"
        f"⚠️ حرف الروي (القافية) لهذا البيت هو: \"{rawi}\"\n"
        f"يجب أن ينتهي **كل شطر ثانٍ** من الأبيات الجديدة بنفس هذا الحرف بالضبط — "
        f"لا تنتقل لحرف روي آخر بعد البيت الأول مهما كان السبب.\n\n"
        f"اكتب استمرارًا جديدًا لهذه القصيدة (٣-٥ أبيات) يبدأ بعد هذا البيت مباشرة، "
        f"مع الحفاظ على الأسلوب والوزن والمعنى المتماسك ونفس حرف الروي بكل الأبيات، "
        f"دون نسخ أي شيء من الأصل."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
