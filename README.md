# Arabic Poetry Intelligence using Semantic Search and Large Language Models

## Overview

This project explores how modern Natural Language Processing (NLP) techniques can be applied to classical Arabic poetry. It combines semantic search, vector databases, and large language models (LLMs) to retrieve relevant poems, generate new poetry in the style of famous poets, and rewrite poems while preserving their original style.

The project is divided into three main tasks:

1. **Semantic Search for Arabic Poetry**
2. **Generate a poem Like a famous Poet**
3. **Rewrite the poem While Keeping Style**

Together, these tasks demonstrate the use of embeddings, vector databases, retrieval-augmented generation (RAG), and prompt engineering for Arabic text generation.

---

# Project Tasks

## Task 1 — Semantic Search

### Objective

Build a semantic search engine capable of retrieving Arabic poems based on their meaning rather than exact keyword matching.

Instead of relying on traditional text search, every poem is converted into a numerical embedding using a multilingual embedding model. Similar poems are then retrieved using vector similarity search.

### Pipeline

Dataset
↓

Preprocessing

↓

Chunking Long Poems

↓

Sentence Embeddings (BAAI/bge-m3)

↓

ChromaDB Vector Database

↓

Semantic Retrieval

### Features

- Semantic similarity search
- Fast retrieval using ChromaDB
- Metadata filtering
- Supports Arabic text
- Retrieves top-k most relevant poems

### Metadata Stored

Each poem stores:

- Title
- Poet
- Era
- Theme
- Source URL

which enables filtering during retrieval.


---

## Task 2 — Generate Like a Poet

### Objective

Generate completely new Arabic poems that imitate the writing style of a selected classical poet while discussing a user-specified topic.

Instead of asking an LLM to imitate a poet directly, the project first retrieves similar poems from the vector database. These poems become examples inside the prompt.

This Retrieval-Augmented Generation (RAG) approach gives the language model stylistic guidance without copying existing poems.

### Workflow

User Input

↓

Select Poet

↓

Select Theme

↓

Semantic Search

↓

Retrieve Similar Poems

↓

Build Prompt

↓

LLM (OpenAI)

↓

Generate New Original Poem

### Example Input

```
Poet : المتنبي

Theme : حكمة

Topic : العلم
```

### Example Output

```

إذا ما ارتقى العلمُ في نفوسِ فتىً *** تاه عن دنسِ الجهلِ كلُّ مَكْرَهِ

وصارَ الأفقُ لهُ منارًا يُضيءُ *** والصبرُ مفتاحُهُ في كلِّ مَفْتَهِ

لا يَذلُّ من حملَ علْمًا على الأعناقِ *** ولو جَرى الدهرُ عليهِ نَحرَهُ وهَتَكِ

يَركبُ المجدَ بحُصانهِ الحكمةِ *** ويرفَعُ رايةَ الفهمِ فوقَ القممِ

ليسَ يطلبُ العلمَ إلا من غوى بِهِ *** أو مُحبٍّ لهُ، فذاك هو الحِكَمُ

فكم عَلَّمَ الدهرُ من جاهلٍ صَبُرًا *** حتى دَجَّنَ في قلبهِ بَذورَهُ الثِّمَرِ

فالعلمُ للبشرِ كالندى للفجرِ *** يوقظُ العقلَ من غفلةِ السِّترِ

فلا تَصغرَنَّ من العلمِ هِمَّةً *** فالعلمُ مرسى النفسِ في السفرِ

وما الحياةُ إلا جَسرٌ عَابِرٌ *** والعلمُ لهُ السُلوكُ والبَصَرُ
إذا ما ارتقى العلمُ في نفوسِ فتىً
تاه عن دنسِ الجهلِ كلُّ مكرهِ

وصارَ الأفقُ لهُ منارًا يُضيءُ
والصبرُ مفتاحُهُ في كلِّ مَفْتَهِ
...
```

The generated poem is inspired by the retrieved examples but is newly generated rather than copied.

---

## Task 3 — Rewrite poem While Keeping Style

### Objective

Rewrite an existing Arabic poem while preserving the stylistic characteristics of the original poet.

Instead of generating from scratch, the language model receives:

- the original poem
- stylistic examples retrieved from the vector database

The model then rewrites the poem while maintaining:

- vocabulary
- poetic tone
- imagery
- writing style

while changing the wording and expressions.

### Workflow

Original Poem

↓

Semantic Retrieval

↓

Retrieve Similar Poems

↓

Prompt Construction

↓

LLM

↓

Rewritten Poem

### Example Input

```

```

### Example Output

```

```

This task demonstrates style transfer using Retrieval-Augmented Generation.

---

# Dataset ( [Arabic Poetry Dataset](https://huggingface.co/datasets/Fatimah8Moheeb/Arabic-Poetry-Dataset/blob/main/full_final_poems.csv) )

The project uses a dataset containing  **128,499 classical Arabic poems**.

Each record includes:

- poem ID
- poet ID
- poem title
- Poem Text
- Poet Name
- era
- theme
- Verses Count
- source URL


---

# Embedding Model

The project uses

```
BAAI/bge-m3
```

because it

- supports multilingual text
- performs well on Arabic
- produces high-quality semantic embeddings
- works efficiently for retrieval tasks

---

# Vector Database ([Chroma Database](https://www.kaggle.com/datasets/nabaanabeeh/arabic-poetry-chroma?select=chroma.sqlite3))

The project uses **ChromaDB** to store

- embeddings
- poem text
- metadata

This enables efficient semantic search with metadata filtering.

---

# Large Language Models

### OpenAI API

GPT models for higher-quality generation.

---

# Technologies Used

- Python
- Pandas
- ChromaDB
- Sentence Transformers
- BAAI/bge-m3
- OpenAI API


---

##  How to Run

1. Clone the repo and open it in VS Code
2. Install dependencies:
   ```
   pip install -r requirements.txt
   pip install flask python-dotenv
   ```
3. Download the merged vector database from Kaggle and unzip it into a folder named `vectordb/` at the project root
4. Add your OpenAI API key to a `.env` file at the project root:
   ```
   OPENAI_API_KEY=your-key-here
   ```
5. Make sure `full_final_poems.csv` is at the project root
6. Run the app:
   ```
   python app.py
   ```
7. Open your browser at `http://127.0.0.1:5000`


---

# Authors

- Najaf Alsabia
- Nabaa Alaswad
- Fatimah Alwarsh
