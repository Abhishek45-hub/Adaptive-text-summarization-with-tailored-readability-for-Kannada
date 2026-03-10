import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from readability import readability_kre

MODEL_PATH = "PATH_TO_MODEL"
MAX_SRC_LEN = 768

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.eval()

def summarize(article: str, max_len=120):
    prompt = "ಸೂಚನೆ: ಕೆಳಗಿನ ಲೇಖನವನ್ನು ಸಂಕ್ಷಿಪ್ತವಾಗಿ ಸಾರಾಂಶ ಮಾಡಿ.\n\nಲೇಖನ:\n" + article
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SRC_LEN)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=max_len, num_beams=4)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text, readability_kre(text)
