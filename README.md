# Adaptive Text Summarization for Kannada

This project trains and runs a Kannada text summarization model with readability-aware prompting.

The training pipeline:
1. Loads Kannada article-summary pairs from a CSV file.
2. Computes a readability score (KRE) from each target summary.
3. Builds an instruction prompt that includes the target readability.
4. Fine-tunes a seq2seq model for summarization.

## Project Structure

```
Adaptive-Text-Summarization-For-Kannada-main/
├── data_utils.py
├── inference.py
├── prompts.py
├── readability.py
├── train.py
├── requirements.txt
├── README.md
├── data/
│   └── dataset.csv
└── model/
	├── adapter_config.json
	├── added_tokens.json
	└── README.md
```

## Requirements

Install dependencies from requirements.txt:

```
torch
transformers
datasets
evaluate
sentencepiece
pandas
numpy
scikit-learn
accelerate
```

## Setup

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset Format

Expected CSV columns are defined in data_utils.py:

- id
- kannada_article
- kannada_highlights

Your current dataset file data/dataset.csv also contains readability_score and readability_level columns, which are acceptable as additional columns.

Rows with empty article or summary text are removed during loading.

## Training

Update these placeholders in train.py before running:

- CSV_PATH = "PATH_TO_FILE"
- output_dir in Seq2SeqTrainingArguments = "PATH_TO_OUTPUT_DIR"
- save_model path = "PATH_TO_OUTPUT_DIR/final"

Default model used for fine-tuning:

- ai4bharat/MultiIndicSentenceSummarizationSS

Run training:

```bash
python train.py
```

What train.py does:

- Computes KRE readability for each target summary.
- Creates input prompt using prompts.py.
- Splits data into train/validation.
- Tokenizes and fine-tunes a seq2seq model.
- Evaluates with ROUGE.
- Saves final model and tokenizer.

## Inference

Update this placeholder in inference.py:

- MODEL_PATH = "PATH_TO_MODEL"

Then use the summarize function.

Example:

```python
from inference import summarize

article = "ನಿಮ್ಮ ಕನ್ನಡ ಲೇಖನವನ್ನು ಇಲ್ಲಿ ನೀಡಿ"
summary, kre = summarize(article)
print(summary)
print("KRE:", kre)
```

The summarize function returns:

- Generated summary text
- Readability score (KRE) for the generated summary

## Readability Module

readability.py provides:

- readability_kre(text): Returns a readability score in range 0-100.
- kre_category(kre): Maps score to Kannada readability levels.

## Notes

- The model/ folder currently includes adapter and tokenizer-related files.
- Files ending with .Zone.Identifier are Windows metadata files and are not required for training/inference.
- If you are using a local fine-tuned model, set MODEL_PATH to the saved final model directory.
