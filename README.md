# Task 2 — Rewrite Poem (Style-Preserving Continuation)

## الفكرة

المستخدم يكتب جزء بسيط من قصيدة يعرفها (مو القصيدة كاملة) + بيت أول جديد يبيه.
النظام:

1. **يلقى القصيدة الأصلية** بسرعة عن طريق البحث بالـ vector DB المشترك (`vector_lookup.py`)
2. **يولّد استمرار جديد** بنفس الوزن والقافية عن طريق GPT (`rewrite_poem.py` + `prompts.py`)

```
مستخدم يكتب جزء  →  find_original_poem()  →  القصيدة الأصلية كاملة
                                                    ↓
مستخدم يكتب بيت أول جديد  →  rewrite_poem()  →  قصيدة جديدة
```

## الملفات

| الملف | الوظيفة |
|---|---|
| `vector_lookup.py` | يبحث بسرعة عن القصيدة الأصلية باستخدام الـ ChromaDB المشترك |
| `rewrite_poem.py` | الدالة الرئيسية `find_original_poem()` + `rewrite_poem()` — هذا اللي يستدعى من `app.py` |
| `prompts.py` | يبني الـ prompt (few-shot) اللي يرسل لـ GPT |
| `requirements.txt` | المكتبات المطلوبة |

## ⚠️ قبل ما يشتغل — لازم تتأكدين من زميلتك (Person 1)

بملف `vector_lookup.py` فيه 3 قيم لازم تطابق بالضبط اللي استخدمتها بـ `build_db.py`:

```python
VECTOR_DB_PATH = "./vectordb"   # مسار المجلد اللي حفظت فيه الـ PersistentClient
COLLECTION_NAME = "poems"        # اسم الـ collection
```

+ لازم يكون embedding model نفسه (اللي استخدمته وقت البناء) هو نفسه المخزون
جوا الـ collection، عشان الاستعلام يستخدم نفس الـ embedding function تلقائيًا.

كمان تأكدي من **أسماء مفاتيح الـ metadata** (`poet_name`, `era`, `title`,
`full_text`) تطابق اللي خزنتها هي بالضبط — إذا فيها فرق بسيط بالاسم
(مثلاً `poet` بدل `poet_name`) لازم تعدلين بـ `find_original_poem_vector()`.

## الرن بدون ما تنتظرين الـ vector DB

لو الـ vector DB ما جاهز بعد، الكود يشتغل تلقائيًا على نسخة احتياطية
(بحث نصي مباشر بالـ CSV) — تقدرين تختبرين شغلك اليوم، وبعدين لما تجهز
هي الـ vector DB، الكود يستخدمه تلقائيًا بدون أي تعديل من طرفك.

## الرن

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="..."   # أو userdata بـ Colab
python rewrite_poem.py
```
