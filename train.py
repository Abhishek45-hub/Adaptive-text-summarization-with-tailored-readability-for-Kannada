#from sentence_transformer import SentenceTransformer

import torch
import numpy as np
import evaluate
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

from data_utils import load_csv, to_dataset
from readability import readability_kre
from prompts import score_prompt

CSV_PATH = "PATH_TO_FILE"
MODEL_NAME = "ai4bharat/MultiIndicSentenceSummarizationSS"

MAX_SRC_LEN = 768
MAX_TGT_LEN = 128

df = load_csv(CSV_PATH)
df["kre"] = df["kannada_highlights"].astype(str).apply(readability_kre)
df["input_text"] = df["kre"].apply(score_prompt) + "\n\nಲೇಖನ:\n" + df["kannada_article"]
df["labels_text"] = df["kannada_highlights"]

ds = to_dataset(df)
split = ds.train_test_split(test_size=0.1, seed=42)
datasets = DatasetDict(train=split["train"], validation=split["test"])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def preprocess(batch):
    inputs = tokenizer(batch["input_text"], truncation=True, max_length=MAX_SRC_LEN)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["labels_text"], truncation=True, max_length=MAX_TGT_LEN)
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized = datasets.map(preprocess, batched=True, remove_columns=datasets["train"].column_names)

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    dp = tokenizer.batch_decode(preds, skip_special_tokens=True)
    dl = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return rouge.compute(predictions=dp, references=dl)

args = Seq2SeqTrainingArguments(
    output_dir="PATH_TO_OUTPUT_DIR",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    evaluation_strategy="steps",
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model),
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("PATH_TO_OUTPUT_DIR/final")
tokenizer.save_pretrained("PATH_TO_OUTPUT_DIR/final")
