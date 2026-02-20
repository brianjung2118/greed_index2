# greed2

Script to sample Naver finance discussion titles, pre-process with soynlp, and save to CSV.

## Setup

Use the project’s virtual environment so the script finds its dependencies (soynlp, pymysql, etc.):

```bash
# From project root (e.g. rii)
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r greed2/requirements.txt
```

Or run with the venv Python without activating: `.venv/bin/python greed2/scripts/gather_preprocess_titles.py`

DB credentials are in the script (same as `greed`); no env vars needed.

## Run

From project root:

```bash
python greed2/scripts/gather_preprocess_titles.py
```

Output: `greed2/data/titles_sample_5000.csv` with columns `original_text`, `preprocessed_text`.

Preprocessing uses soynlp’s `RegexTokenizer` only; repeated punctuations, words, expressions, and emojis are not removed.

---

## Label titles with greed score (0–4) via LLM

After `titles_sample_5000.csv` exists, you can label each preprocessed title with an ordinal greed score using a Hugging Face LLM:

- **0** = extreme fear  
- **1** = fear / bearish  
- **2** = neutral  
- **3** = greed / bullish  
- **4** = extreme greed  

Models: **Qwen3-8B** or **Llama-3.1-8B-Instruct** (choose with `--model`).

```bash
# Use Qwen3-8B (default)
python greed2/scripts/label_greed_llm.py --model qwen3-8b

# Use Llama-3.1-8B-Instruct (may require Hugging Face login for gated model)
python greed2/scripts/label_greed_llm.py --model llama-3.1-8b
```

The script adds a **`greed_label`** column to `greed2/data/titles_sample_5000.csv`. It checkpoints every 100 rows (`--save-every 100`); if interrupted, run again and it will skip already-labeled rows. Optional: `--limit N` to label only N rows, `--csv path` for a different file.
