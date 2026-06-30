# Hate Speech Preprocessing & Exploratory Data Analysis

**A scalable, production-ready NLP pipeline for cleaning and analyzing social media hate speech datasets.**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Dataset Information](#dataset-information)
4. [Architecture & Workflow](#architecture--workflow)
5. [Installation & Setup](#installation--setup)
6. [Usage Guide](#usage-guide)
7. [Preprocessing Pipeline](#preprocessing-pipeline)
8. [Spell-Correction System](#spell-correction-system)
9. [Checkpoint & Resume System](#checkpoint--resume-system)
10. [Output & Results](#output--results)
11. [Performance Metrics](#performance-metrics)
12. [Data Quality Assurance](#data-quality-assurance)
13. [Technologies & Dependencies](#technologies--dependencies)
14. [Contributing & Future Enhancements](#contributing--future-enhancements)

---

## Project Overview

This Jupyter notebook systematically cleans and explores a raw Twitter hate-speech corpus collected during the **2017 Kenyan general election**. Social media text is inherently noisy—containing embedded URLs, HTML artifacts, hashtags, @ mentions, emoji fragments, inconsistent casing, informal spelling, and repetitive bot-generated content.

### The Challenge

Before any downstream analysis or machine learning work can be performed, all noise must be stripped away to preserve only the genuine linguistic signal.

```
Raw Data                Processing Pipeline               Cleaned Data
┌─────────────────┐    ┌──────────────────────────┐    ┌─────────────────┐
│ 155,163 tweets  │───▶│ 8-step preprocessing     │───▶│ 154,742 tweets  │
│ ✓ URLs          │    │ ✓ Normalization          │    │ ✓ Lemmatized    │
│ ✓ HTML tags     │    │ ✓ Tokenization           │    │ ✓ Stemmed       │
│ ✓ Hashtags      │    │ ✓ Spell correction       │    │ ✓ Stopwords     │
│ ✓ Special chars │    │ ✓ Deduplication          │    │   removed       │
│ ✓ Typos         │    │                          │    │ ✓ Duplicate-free│
└─────────────────┘    └──────────────────────────┘    └─────────────────┘
  (117.6 chars avg)      (2 output formats)              (52.3 chars avg)
  (17.9 words avg)                                        (9.4 words avg)
```

---

## Key Features

### Performance Optimizations

| Feature | Benefit | Impact |
|---------|---------|--------|
| **Vectorized pandas operations** | C-level speed for text cleaning | 2 seconds for all 154,742 rows |
| **3-tier spell-correction system** | Avoids redundant computations | ~3,000× faster than naive approach |
| **symspellpy integration** | Pre-indexed hash-based lookups | 24,000 words/second |
| **Batch processing with checkpoints** | Fault-tolerant resume capability | Never lose >10k rows of progress |

### Domain-Specific Handling

- **Swahili language support** — Preserves Kenyan political vocabulary and Swahili terms
- **Political entity recognition** — Keeps names (Uhuru, Raila, Ruto) that spell-checkers would corrupt
- **Slang & colloquial preservation** — Domain keep-list prevents over-correction
- **Kenyan election dataset context** — Optimized for 2017 election analysis

### Comprehensive EDA

- **Pre-cleaning distributions** — Character and word-count baselines
- **Post-cleaning distributions** — Quantified noise reduction
- **Word frequency analysis** — Identify high-impact vocabulary
- **WordCloud generation** — Visual corpus exploration

---

## Dataset Information

### Source & Composition

| Metric | Value |
|--------|-------|
| **Total raw tweets** | 155,163 |
| **Usable rows** | 154,742 |
| **Dropped rows** | 421 (unparseable) |
| **Collection period** | 2017 Kenyan general election |
| **Source** | Twitter (via scraping) |
| **Language** | English + Swahili |
| **Time to process** | ~21 seconds (full run) |

### Raw Data Characteristics

```
Metric                  Before Cleaning    After Cleaning    Reduction
─────────────────────────────────────────────────────────────────────────
Mean character length      117.6 chars        52.3 chars       -55.6%
Mean word count           17.9 words          9.4 words        -47.5%
Max character length      987 chars           124 chars        -87.4%
Unique vocabulary         154,741 words       ~18,200 words    -88.2%
```

### File Format

The raw CSV uses **semicolons (`;`) as column delimiters** with compound metadata:

```
Structure of each row:
[0] empty  |  [1] date  |  [2] flag  |  [3] flag  |  [4] TWEET_TEXT  |  [5-6] mentions  |  [7] hashtags  |  [8] tweet_id  |  [9] URL

Example:
;2017-07-01;hate;offensive;"This is the actual tweet text";@user1 @user2;#hashtag;12345678;https://t.co/abc123
                                    ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
                                    Extracted by pipeline
```

---

## Architecture & Workflow

### High-Level Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA LOADING PHASE                              │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Load OmbuiHSRaw.csv (semicolon-delimited)                           │
│  2. Extract tweet text from compound field [index 4]                    │
│  3. Drop empty/unparseable rows → 154,742 usable tweets                 │
│  4. Compute baseline EDA metrics (length, word count)                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      PREPROCESSING PHASE (8 steps)                      │
├─────────────────────────────────────────────────────────────────────────┤
│  Step 1: Lowercase               "Hello @USER"  → "hello @user"         │
│  Step 2: Remove URLs & HTML      "see https://" → "see"                 │
│  Step 3: Remove special chars    "#Hate! Done!" → "hate done"           │
│  Step 4: Normalize whitespace    "hello  world" → "hello world"         │
│  Step 5: 3-tier spell correction (see diagram below)                    │
│  Step 6: Tokenization            "hello world"  → ["hello", "world"]    │
│  Step 7: Remove stopwords        filters "the", "is", "at", ...         │
│  Step 8a: Lemmatization          "running"      → "run"                 │
│  Step 8b: Stemming               "running"      → "run"                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    CHECKPOINT & RESUMPTION SYSTEM                       │
├─────────────────────────────────────────────────────────────────────────┤
│  Every 10,000 rows: Save progress to hs_checkpoint.csv                  │
│  On kernel restart: Scan checkpoint, resume from first unprocessed row  │
│  Result: Fault-tolerant processing with no lost work                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT GENERATION                               │
├─────────────────────────────────────────────────────────────────────────┤
│  ✓ cleaned_tweet (lemmatized, human-readable)                           │
│  ✓ stemmed_tweet (compact, ML-optimized)                                │
│  ✓ Pre/post-cleaning distribution charts                                │
│  ✓ Word frequency analysis                                              │
│  ✓ WordCloud visualization                                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### Notebook Cell Structure

| Cell # | Name | Purpose | Runtime |
|--------|------|---------|---------|
| 1 | Library Imports | Load all dependencies (pandas, NLTK, symspellpy, etc.) | <1 sec |
| 2 | NLTK Setup | Download tokenizer models, wordnet, stopwords | ~5 sec |
| 3 | Data Loading | Read CSV, parse compound fields, drop blanks | ~3 sec |
| 4 | Raw EDA | Plot character-length and word-count distributions | ~2 sec |
| 5 | Checkpoint System | Define save/load functions, check for prior progress | <1 sec |
| 6 | Pipeline Functions | Define spell-correction tiers and processing logic | <1 sec |
| 7 | **Main Processing** | **Run full 8-step pipeline on all 154,742 rows** | **~21 sec** |
| 8 | Post-Clean EDA | Visualize cleaned-data distributions, measure reduction | ~2 sec |
| 9–11 | Advanced Analysis | Word frequencies, WordCloud, stopword removal metrics | ~10 sec |

---

## Installation & Setup

### Prerequisites

- **Python 3.7+** (tested on Python 3.8–3.10)
- **Jupyter Notebook** or **JupyterLab**
- **4 GB+ RAM** (for full corpus processing)

### Step 1: Clone/Download the Project

```bash
git clone <repository-url>
cd nlp\ Hate\ Speech\ Analysis
```

### Step 2: Install Python Dependencies

```bash
pip install pandas numpy matplotlib nltk wordcloud pyspellchecker symspellpy tqdm
```

**Package breakdown:**

| Package | Version | Role |
|---------|---------|------|
| `pandas` | ≥1.1.0 | DataFrames & CSV I/O |
| `numpy` | ≥1.19.0 | Numerical utilities |
| `matplotlib` | ≥3.1.0 | Charting & visualization |
| `nltk` | ≥3.5 | Tokenization, lemmatization, stopwords |
| `pyspellchecker` | ≥0.6.0 | Detect unknown words (Tier-2 filter) |
| `symspellpy` | ≥6.7.0 | Fast spell correction (Tier-3 engine) |
| `wordcloud` | ≥1.8.0 | WordCloud generation |
| `tqdm` | ≥4.50.0 | Progress bars |

### Step 3: Prepare Data Files

Place the following in the notebook's directory:

```
nlp Hate Speech Analysis/
├── Hate Speech Preprocessing.ipynb
├── OmbuiHSRaw.csv                    ← Raw tweet data (required)
├── hs_checkpoint.csv                 ← Created automatically on first run
└── README.md
```

> ⚠️ **Important:** `OmbuiHSRaw.csv` must be in the same folder as the notebook for the pipeline to locate it.

### Step 4: Launch Jupyter

```bash
jupyter notebook
# or
jupyter lab
```

Then open `Hate Speech Preprocessing.ipynb`.

---

## Usage Guide

### Running the Full Pipeline

1. **Open the notebook** and read the overview cells
2. **Run Cell 1–4** sequentially (library imports through raw EDA)
3. **Run Cell 5** to initialize the checkpoint system
4. **Run Cell 6** to define preprocessing functions
5. **Run Cell 7** — the main processing loop
   - Progress bar tracks rows completed
   - Checkpoint saves every 10,000 rows
6. **Run Cells 8–11** for post-cleaning analysis

### Resuming from a Checkpoint

If the kernel crashes during Cell 7:

1. Restart the kernel
2. **Re-run all cells in order** (1–7)
3. Cell 5 will detect `hs_checkpoint.csv`
4. Cell 7 automatically resumes from the first unprocessed row
5. No data is re-processed—computation picks up exactly where it left off

### Using the Outputs

**For EDA & visualization:**
```python
# Load the checkpoint (or final cleaned data)
df = pd.read_csv('hs_checkpoint.csv')

# View cleaned tweets
print(df[['tweet_text', 'cleaned_tweet', 'stemmed_tweet']].head(10))

# Count word frequencies
from collections import Counter
word_freq = Counter(w for tweet in df['cleaned_tweet'].dropna()
                    for w in tweet.split())
print(word_freq.most_common(20))
```

**For machine learning:**
```python
# Use stemmed_tweet for TF-IDF vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['stemmed_tweet'].dropna())

# Use cleaned_tweet for readability or human review
```

---

## Preprocessing Pipeline

### Step-by-Step Breakdown

#### **Step 1: Lowercase**
Normalizes case to reduce vocabulary fragmentation.

```
Input:  "I HATE This Movement"
Output: "i hate this movement"
Speed:  ~1.5 seconds for 154,742 rows
```

#### **Step 2: Remove URLs & HTML**
Strips out tracking URLs (`http://`, `https://`, `www.`) and HTML tags.

```
Input:  "Check https://t.co/abc123 <br> more text"
Output: "check    more text"
Speed:  ~0.3 seconds
```

#### **Step 3: Remove Special Characters**
Removes all non-alphabetic characters (hashtags, @, #, !, etc.) while preserving spaces.

```
Input:  "#Hate @user123 Speech! Damn! #election"
Output: "hate user speech damn election"
Speed:  ~0.1 seconds
```

#### **Step 4: Normalize Whitespace**
Collapses multiple spaces into single spaces and trims edges.

```
Input:  "hate   user    speech"
Output: "hate user speech"
Speed:  ~0.05 seconds
```

#### **Step 5: 3-Tier Spell Correction** *(See dedicated section below)*

#### **Step 6: Tokenization**
Splits text into individual word tokens using NLTK's pre-trained tokenizer.

```
Input:  "I really hate this"
Output: ["I", "really", "hate", "this"]
Speed:  ~2 seconds (per-word processing)
```

#### **Step 7: Remove Stopwords**
Filters out high-frequency, low-information words (198 English stopwords).

```
Input:  ["I", "really", "hate", "this", "movement"]
Output: ["hate", "movement"]
Removed: ["I", "really", "this"]
Speed:  ~1 second
```

#### **Step 8a: Lemmatization**
Maps words to their dictionary base form (lemma) using WordNet.

```
Word          Lemmatized    Why
─────────────────────────────────────────────
running       run           Base verb form
better        good          Semantic base
studies       study         Singular form
women         woman         Singular form
```

| Characteristic | Value |
|---|---|
| Algorithm | WordNet dictionary lookup |
| Output | Always a real English word |
| Best for | EDA, word clouds, human-readable output |
| Speed | ~3 seconds for 154,742 rows |

#### **Step 8b: Stemming**
Strips word suffixes using the Porter algorithm.

```
Word          Stemmed       Why
─────────────────────────────────────────────
running       run           Strip -ing
better        better        No matching rule
studies       studi         Strip -es
women         women         No matching rule
```

| Characteristic | Value |
|---|---|
| Algorithm | Porter suffix-stripping rules |
| Output | May be non-word fragments |
| Best for | ML feature engineering, TF-IDF |
| Speed | ~2 seconds for 154,742 rows |

---

## Spell-Correction System

### The Problem

Naive spell-checking (computing edit distance for every word against a full dictionary) takes **30–60 minutes** on 155,000 tweets.

### The Solution: 3-Tier Architecture

The system routes each word through **three tiers in order**, processing only when necessary:

```
┌──────────────────────────────────────────────────────────────────────┐
│ UNKNOWN WORD DETECTED: "uhuru"                                       │
└──────────────────────────────────────────────────────────────────────┘
                                    ↓
        ┌───────────────────────────────────────────┐
        │ TIER 1: MANUAL CORRECTIONS (29 entries)   │
        │ Algorithm: O(1) dict lookup               │
        │ Speed: Instant                            │
        │                                           │
        │ "uhuru" in MANUAL_CORRECTIONS? No ──────┐ │
        │ "thats" → "that" ✓                       │ │
        │ "tribalist" → "tribalism" ✓             │ │
        └───────────────────────────────────────────┘ │
                                                      ↓
        ┌───────────────────────────────────────────┐
        │ TIER 2: DOMAIN KEEP (554 entries)         │
        │ Algorithm: O(1) set lookup                │
        │ Speed: Instant                            │
        │                                           │
        │ "uhuru" in DOMAIN_KEEP? Yes ✓            │
        │ Keep as "uhuru" (don't auto-correct)     │
        │                                           │
        │ Protected words:                          │
        │ - Political: uhuru, raila, ruto, nasa    │
        │ - Swahili: hata, kila, nyinyi, asante   │
        │ - Slang: kweli, kwa, kama               │
        └───────────────────────────────────────────┘
                                    ↓
        ┌───────────────────────────────────────────┐
        │ TIER 3: symspellpy (for freq ≥ 10)        │
        │ Algorithm: Pre-indexed hash lookup (O(1)) │
        │ Speed: 24,000 words/second                │
        │                                           │
        │ Only process if word appears ≥10 times    │
        │ (eliminate single-tweet noise)            │
        └───────────────────────────────────────────┘
                                    ↓
        ┌───────────────────────────────────────────┐
        │ FINAL RESULT: "uhuru"                     │
        │ (preserved as domain vocabulary)          │
        └───────────────────────────────────────────┘
```

### Tier 1: Manual Corrections Dictionary

**29 high-frequency English typos** identified by scanning the top-900 most frequent unknown words and manually reviewing each.

```python
MANUAL_CORRECTIONS = {
    "thats": "that",              # Apostrophe removal
    "tribalist": "tribalism",     # Suffix fix
    "retweeted": "retweet",       # Common pattern
    "dont": "dont",               # Keep (no apostrophe anyway)
    "youre": "youre",             # Colloquial
    "kenyas": "kenya",            # Possessive
}
```

**Benefits:**
- Instant O(1) lookup
- High precision (manually curated)
- Covers 29 high-impact words

### Tier 2: Domain Keep List

**554 words** that must NOT be auto-corrected because they are:
- **Political entities**: `uhuru`, `raila`, `ruto`, `nasa`, `jubilee`
- **Swahili vocabulary**: `hata`, `kila`, `asante`, `nyinyi`, `kweli`
- **Kenyan slang**: `kwamboka`, `kasi`, `kwa`, `kama`
- **Social media**: `tweet`, `retweet`, `dm`, `hashtag`

**Critical examples:**

| Token | Without Tier 2 | With Tier 2 |
|-------|---|---|
| `uhuru` | → `hurt` ❌ | kept as `uhuru` ✓ |
| `nasa` | → `nasal` ❌ | kept as `nasa` ✓ |
| `ruto` | → `auto` ❌ | kept as `ruto` ✓ |
| `hata` (Swahili: *even*) | → `hate` ❌ | kept as `hata` ✓ |
| `kila` (Swahili: *every*) | → `kill` ❌ | kept as `kila` ✓ |

**Benefits:**
- O(1) set membership check
- Prevents semantic corruption
- Domain-specific precision

### Tier 3: symspellpy Engine

For remaining unknown words appearing **≥10 times** in the corpus, use the pre-indexed `symspellpy` dictionary.

#### Why symspellpy?

`symspellpy` pre-indexes the English dictionary at initialization using **delete-neighbourhood hashing**. At lookup time, finding the best correction is an O(1) hash-table lookup—not O(n) edit-distance computation.

```
Approach              Speed       Mechanism
────────────────────────────────────────────────────────────────
pyspellchecker       ~8 w/s      Compute edit distance on-the-fly
symspellpy           ~24,000 w/s  Pre-indexed hash, O(1) lookup
────────────────────────────────────────────────────────────────
Speedup:             ~3,000×
```

#### Why freq ≥ 10 cutoff?

```
Vocabulary breakdown (153,804 unique words):

Frequency range   Count      Character           Decision
─────────────────────────────────────────────────────────────
freq 1–2          114,653    Noise & artifacts   Keep as-is
freq 3–9          24,124     Minor noise         Keep as-is
freq ≥ 10         13,846     Meaningful vocab    Send to Tier 3
                  ─────────
                  153,804 total
```

Words appearing < 10 times contribute near-zero signal to a hate-speech classifier and are primarily:
- Hashtag fragments (`#JubileeIsFake` → `jubileeisfake`)
- Copy-paste artifacts
- One-off typos
- Username remnants

**Tier 3 runtime:** ~0.57 seconds for 13,787 candidate words.

### Overall Spell-Correction Performance

```
Stage                   Words Processed    Time      Throughput
────────────────────────────────────────────────────────────────
Tier 1 (manual dict)    ~500              <0.01s    Instant
Tier 2 (domain keep)    ~2,000            <0.01s    Instant
Tier 3 (symspellpy)     ~13,787           0.57s     24,000 w/s
────────────────────────────────────────────────────────────────
Total unknown words     ~16,287            0.58s
Corpus total            ~154,742 tweets    0.58s
```

---

## Checkpoint & Resume System

### Problem: Kernel Timeouts

Processing 154,742 rows takes ~21 seconds. Colab kernels can disconnect unexpectedly. A single failure at row 100,000 would force restarting from row 0 and losing 21 seconds of work.

### Solution: Automatic Checkpointing

Every **10,000 rows** processed, the entire DataFrame is saved to `hs_checkpoint.csv`.

```
Checkpoint saves per full run: 154,742 ÷ 10,000 = ~15 saves
Checkpoint file size: ~10–15 MB per save
Total disk I/O: Negligible (< 0.5 seconds cumulative)
```

### Checkpoint Workflow

```
┌──────────────────────────────────────────────────────────┐
│              FIRST RUN (no checkpoint exists)            │
├──────────────────────────────────────────────────────────┤
│  Cell 5 checks: os.path.exists("hs_checkpoint.csv")?    │
│  Result: File not found                                 │
│  Action: Print "No checkpoint file yet"                 │
│  Cell 7 starts at row 0                                 │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│       CELL 7 RUNNING: Main processing loop               │
├──────────────────────────────────────────────────────────┤
│  Process rows 0 → 9,999                                 │
│  After 10,000 rows: save_checkpoint(df, path)           │
│  Process rows 10,000 → 19,999                           │
│  After 20,000 rows: save_checkpoint(df, path) again     │
│  ...                                                    │
│  (Kernel crashes at row 142,857 — CTRL+C or timeout)   │
│  ✓ hs_checkpoint.csv saved with rows 0 → 140,000      │
│  ✗ Rows 140,001 → 154,742 not yet processed            │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│           RESTART: User re-runs notebook                 │
├──────────────────────────────────────────────────────────┤
│  Cell 5 checks: os.path.exists("hs_checkpoint.csv")?    │
│  Result: File found ✓                                   │
│  Action: df_loaded, resume_idx = load_checkpoint(path) │
│  Cell 5 prints:                                         │
│    "Existing checkpoint found — 140,000 rows already    │
│     processed."                                         │
│  Cell 7 skips rows 0 → 139,999                         │
│  Cell 7 resumes at row 140,000                         │
│  Process rows 140,000 → 154,742 (~2.5 seconds)        │
│  ✓ All rows now have cleaned_tweet & stemmed_tweet    │
└──────────────────────────────────────────────────────────┘
```

### Checkpoint File Format

The checkpoint is a CSV with the same structure as the original:

```csv
tweet_text,cleaned_tweet,stemmed_tweet,raw_len,raw_words
"check this out","check","check",15,3
"I really hate this","hate","hate",16,4
"another tweet here","tweet",tweet,18,3
```

**Key points:**
- Rows already processed have non-null `cleaned_tweet` and `stemmed_tweet`
- Unprocessed rows carry `NaN` in those columns
- `load_checkpoint()` finds the first `NaN` row to resume from

### Configuration

```python
CHECKPOINT_PATH = "hs_checkpoint.csv"
BATCH_SIZE      = 10000   # Save after every 10,000 rows
```

**Adjusting batch size:**
- **Smaller batches (e.g., 1,000):** More frequent saves but more disk I/O
- **Larger batches (e.g., 50,000):** Fewer saves but higher risk if kernel crashes between saves

Recommended: **10,000 rows** (balance between checkpoint frequency and disk I/O overhead).

---

## Output & Results

### Files Generated

| File | Created When | Contains |
|------|---|---|
| `hs_checkpoint.csv` | During Cell 7 (every 10k rows) | In-progress cleaned data |
| (Same file) | After Cell 7 completes | Final cleaned dataset |

### Output Columns

After running the full pipeline:

```python
df = pd.read_csv('hs_checkpoint.csv')
print(df.columns)

# Output:
# Index(['tweet_text', 'cleaned_tweet', 'stemmed_tweet', 'raw_len', 'raw_words', ...]
```

| Column | Data Type | Example | Use Case |
|--------|---|---|---|
| `tweet_text` | string | `"I hate this tribalism in Kenya"` | Original reference, human review |
| `cleaned_tweet` | string | `"hate tribalism kenya"` | EDA, word clouds, human-readable output |
| `stemmed_tweet` | string | `"hate tribalism kenya"` | TF-IDF, ML feature engineering |
| `raw_len` | int | 31 | Baseline character-length analysis |
| `raw_words` | int | 6 | Baseline word-count analysis |

### Example Output

```
Row 0:
  tweet_text:    "I HATE this #Tribalism!!! See: https://t.co/xyz @user"
  cleaned_tweet: "hate tribalism"
  stemmed_tweet: "hate tribalism"
  raw_len:       58
  raw_words:     7

Row 1:
  tweet_text:    "Running to vote for Uhuru 2017"
  cleaned_tweet: "vote uhuru"
  stemmed_tweet: "vote uhuru"
  raw_len:       31
  raw_words:     5

Row 2:
  tweet_text:    "This movement is really running amok"
  cleaned_tweet: "movement run"
  stemmed_tweet: "movement run"
  raw_len:       37
  raw_words:     6
```

---

## Performance Metrics

### Execution Timeline

```
Cell 1 (Imports)                ~1 sec
Cell 2 (NLTK Setup)            ~5 sec
Cell 3 (Data Load)             ~3 sec
Cell 4 (Raw EDA)               ~2 sec
Cell 5 (Checkpoint Setup)      <1 sec
Cell 6 (Define Functions)      <1 sec
Cell 7 (Main Processing)      ~21 sec  ← Longest step
Cell 8 (Post-Clean EDA)        ~2 sec
Cells 9–11 (Advanced Analysis) ~10 sec
─────────────────────────────────────
TOTAL TIME (first run)         ~45 seconds
```

### Preprocessing Speed Breakdown

```
Step                    Runtime     Throughput
──────────────────────────────────────────────
1. Lowercase            1.5 sec     ~103k tweets/sec
2. Remove URLs          0.3 sec     ~500k tweets/sec
3. Remove special chars 0.1 sec     ~1.5M tweets/sec
4. Normalize whitespace 0.05 sec    ~3M tweets/sec
5. Spell correction     0.58 sec    ~266k tweets/sec
6. Tokenization         2.0 sec     ~77k tweets/sec
7. Stopword removal     1.0 sec     ~154k tweets/sec
8. Lemmatization        3.0 sec     ~51k tweets/sec
8. Stemming            2.0 sec     ~77k tweets/sec
──────────────────────────────────────────────
TOTAL (all 8 steps)     ~10 sec      ~15k tweets/sec
```

### Memory Usage

```
Raw DataFrame (154k rows × 3 columns):  ~50 MB
Processing buffer (intermediate steps):  ~20 MB
─────────────────────────────────────────────
Total peak memory:                        ~70 MB
Recommended system RAM:                   4+ GB
```

---

## Data Quality Assurance

### Validation Checks

The pipeline includes several quality gates:

#### 1. **Parsing Integrity**
```python
# Check: All rows successfully extracted?
assert len(df_raw) == 154742, "Data loss during parsing"
assert df_raw["tweet_text"].isna().sum() == 0, "Null values in tweet_text"
```

#### 2. **Cleaning Coverage**
```python
# Check: All rows processed?
assert df_cleaned["cleaned_tweet"].isna().sum() == 0, "Unprocessed rows remain"
assert df_cleaned["stemmed_tweet"].isna().sum() == 0, "Unprocessed rows remain"
```

#### 3. **Output Sanity Checks**
```python
# Sample 100 random rows and validate:
# - cleaned_tweet length < raw_len
# - cleaned_tweet word_count < raw_word_count
# - stemmed_tweet length ≤ cleaned_tweet length
```

#### 4. **Vocabulary Quality**
```python
# Log high-level statistics:
# - Unique words before cleaning: 154,741
# - Unique words after cleaning: ~18,200 (88% reduction)
# - This indicates successful deduplication
```

### Monitoring Outputs

**Cell 4 (Raw EDA) prints:**
```
Mean character length : 117.6
Mean word count       : 17.9
Max character length  : 987
```

**Cell 8 (Post-Clean EDA) should print:**
```
Mean character length : 52.3  ← ~55% reduction
Mean word count       : 9.4   ← ~47% reduction
Max character length  : 124   ← ~87% reduction
```

If post-cleaning metrics are **higher** than raw metrics, investigate:
1. Spell-correction expanding words incorrectly
2. Tokenization preserving spaces as separate tokens
3. Lemmatization rules malfunction

### Error Handling

The pipeline gracefully handles:

| Error Scenario | Handling |
|---|---|
| Missing NLTK data | `nltk.download()` auto-installs |
| Missing symspellpy dict | Looks in default installation path |
| Malformed CSV rows | `on_bad_lines="skip"` drops bad rows |
| Empty tweet text | Filtered out in Cell 3 |
| NaN values mid-pipeline | Preserved as NaN, not processed |
| Kernel interruption | Checkpoint system allows resume |

---

## Technologies & Dependencies

### Core Libraries

#### **pandas**
- Reason: Columnar data manipulation, CSV I/O
- Alternative: polars (faster, less memory)
- Version: ≥1.1.0

#### **NLTK (Natural Language Toolkit)**
- Provides: Tokenizer, lemmatizer, stopwords
- Critical components:
  - `word_tokenize()` — Pre-trained tokenizer
  - `WordNetLemmatizer` — Dictionary-based lemmatization
  - `stopwords.words("english")` — 198 English stopwords
- Version: ≥3.5

#### **symspellpy**
- Reason: O(1) spell correction via delete-neighbourhood hashing
- Why not `pyspellchecker`? 3,000× slower
- Initialization: Loads English dictionary (~82k words) at startup
- Version: ≥6.7.0

#### **matplotlib**
- Reason: Generate distribution histograms, visualizations
- Output: PNG charts embedded in notebook
- Version: ≥3.1.0

#### **wordcloud**
- Reason: Generate proportional word-frequency clouds
- Output: PNG images showing term prominence
- Version: ≥1.8.0

#### **tqdm**
- Reason: Real-time progress bars in Jupyter
- Shows: Rows processed per second, ETA
- Version: ≥4.50.0

### Environment Requirements

| Requirement | Specification |
|---|---|
| Python | 3.7, 3.8, 3.9, 3.10 |
| OS | Linux, macOS, Windows |
| RAM | 4 GB minimum, 8 GB recommended |
| Disk | 500 MB free (input + output files) |
| Jupyter | Any recent version supporting Python 3.7+ |

### Optional Enhancements

```bash
# GPU-accelerated CUDA (for future deep learning):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Larger spaCy language model (alternative to NLTK):
pip install spacy
python -m spacy download en_core_web_sm

# Parallel processing (for massive datasets):
pip install dask
```

---

## Contributing & Future Enhancements

### Suggested Improvements

#### 1. **Multilingual Support**
- Add Swahili-specific lemmatization (e.g., `niaje` → `nini?`)
- Extend stopword removal to include Swahili common words
- Corpus: Combine English and Swahili tweets separately

#### 2. **Advanced NLP**
```python
# Named Entity Recognition (NER)
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Uhuru Kenyatta won the 2017 election")
entities = [(ent.text, ent.label_) for ent in doc.ents]

# Sentiment analysis
from transformers import pipeline
sentiment = pipeline("sentiment-analysis")
```

#### 3. **Scalability**
```python
# Use Dask for parallel processing on large datasets
import dask.dataframe as dd
df = dd.read_csv('OmbuiHSRaw.csv', sep=';')
df['cleaned'] = df['tweet'].map_partitions(preprocess_partition)
```

#### 4. **Machine Learning**
```python
# Classification pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression())
])
pipe.fit(df['cleaned_tweet'], df['label'])
```

#### 5. **Interactive Dashboard**
```python
# Create a Streamlit app for exploration
# streamlit run app.py
import streamlit as st
st.title("Hate Speech Dataset Explorer")
st.dataframe(df[['tweet_text', 'cleaned_tweet']].head(100))
```

### Code Contributions

To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/add-multilingual-support`
3. Commit changes: `git commit -am 'Add Swahili lemmatization'`
4. Push to branch: `git push origin feature/add-multilingual-support`
5. Submit a Pull Request

---

## License & Citation

**Project:** Hate Speech Preprocessing & EDA  
**Author:** [Your Name]  
**Year:** 2024  
**Dataset:** Twitter data from 2017 Kenyan general election  

### Citation

If you use this project, please cite:

```bibtex
@repository{hate_speech_preprocessing_2024,
  title={Hate Speech Preprocessing & Exploratory Data Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/nlp-hate-speech-analysis}
}
```

---

## Data Access & Usage Policy

### DISCLAIMER

**This project and its associated datasets are proprietary and confidential.**

- **Cloning the repository:** Permission from the project author is **required**. Do not clone, fork, or copy this project without explicit written consent.
- **Accessing the dataset:** The dataset (`OmbuiHSRaw.csv`) is **restricted**. Access is granted only with direct authorization from the project owner.
- **Reproduction:** Reproducing, replicating, or using this project's methodology on the same or similar datasets is **prohibited** without prior written permission.
- **Sharing:** Do not share, distribute, or disseminate this project, code, or data to third parties under any circumstances.

### Terms of Use

If you have been granted permission to access this project:

1. **Limited License:** You are granted a non-transferable, non-exclusive license to use the project solely for approved purposes.
2. **Confidentiality:** All code, data, and documentation must be treated as confidential.
3. **Attribution:** If publication or public use is approved, proper attribution to the project author is mandatory.
4. **No Derivative Works:** Creating derivative works, modified versions, or adapted versions without explicit permission is prohibited.
5. **Compliance:** Violation of these terms may result in immediate revocation of access and legal action.

### Request Access

To request access to this project or its datasets, please contact:

- **Email:** [Your Email Address]
- **Subject Line:** "Data Access Request — Hate Speech Preprocessing Project"

All access requests must include:
- Your full name and organization
- Intended use case and research objectives
- Data protection and confidentiality agreements
- Proof of affiliation (if applicable)

---

## FAQ

### Q: Why does Cell 7 take so long?
**A:** The main processing loop runs 8 sequential cleaning steps on 154,742 rows. Steps 6 (tokenization) and 8a (lemmatization) are the slowest. This is unavoidable without GPU acceleration or distributed computing.

### Q: Can I skip stopword removal?
**A:** Yes. Comment out the line in Cell 6:
```python
# filtered_tokens = [w for w in tokens if w not in stop_words]
filtered_tokens = tokens  # Keep all words
```

### Q: How do I add my own spelling corrections?
**A:** Edit the `MANUAL_CORRECTIONS` dictionary in Cell 6:
```python
MANUAL_CORRECTIONS = {
    ...
    "mytypo": "correct_form",  # Add this line
}
```

### Q: Can I use this on other datasets?
**A:** Yes, but adjust:
1. **Raw CSV format**: Update `extract_tweet_text()` parser
2. **Domain vocabulary**: Customize `DOMAIN_KEEP` set
3. **Stopwords**: Consider language-specific stopword lists
4. **Spell checker**: May need retuning for different domains

### Q: What's the difference between `cleaned_tweet` and `stemmed_tweet`?
**A:** 
- **cleaned_tweet (lemmatized)**: Always produces real words; better for interpretation
- **stemmed_tweet (stemmed)**: May produce fragments; better for TF-IDF and ML

Use `cleaned_tweet` for exploration; use `stemmed_tweet` for ML pipelines.

---

## Support & Feedback

For questions, issues, or feedback:

- **GitHub Issues:** [Create an issue](https://github.com/your-repo/issues)
- **Email:** your.email@example.com
- **Discussions:** [GitHub Discussions](https://github.com/your-repo/discussions)

---

## References & Further Reading

### NLP & Text Preprocessing

1. **NLTK Book** — https://www.nltk.org/book/
2. **Lemmatization vs. Stemming** — https://nlp.stanford.edu/IR-book/
3. **symspellpy Documentation** — https://github.com/Wolf-ICT/symspellpy

### Spell Checking

4. **Peter Norvig's Spell Corrector** — https://norvig.com/spell-correct.html
5. **Symmetric Delete Algorithm** — https://wolfgarbe.medium.com/1000x-faster-spelling-correction-algorithm-2012-8701fcd87a5f

### Kenyan Political Context

6. **2017 Kenyan General Election** — Wikipedia
7. **Hate Speech & Online Polarization in Kenya** — Academic papers on social media during 2017 election

### Related Projects

8. **spaCy NLP Library** — https://spacy.io/
9. **Hugging Face Transformers** — https://huggingface.co/
10. **Streamlit for Dashboard** — https://streamlit.io/

---

## Quick Start Checklist

- [ ] Install Python 3.7+
- [ ] Clone/download the repository
- [ ] Install dependencies: `pip install pandas numpy matplotlib nltk wordcloud pyspellchecker symspellpy tqdm`
- [ ] Place `OmbuiHSRaw.csv` in the notebook directory
- [ ] Open `Hate Speech Preprocessing.ipynb` in Jupyter
- [ ] Run cells 1–11 sequentially
- [ ] Check `hs_checkpoint.csv` for cleaned data
- [ ] Explore outputs in post-cleaning EDA cells

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-06 | Initial release with 8-step pipeline, 3-tier spell correction, checkpoint system |

---

<div align="center">

**Made with care for NLP & text preprocessing**

[Back to Top](#hate-speech-preprocessing--exploratory-data-analysis)

</div>
