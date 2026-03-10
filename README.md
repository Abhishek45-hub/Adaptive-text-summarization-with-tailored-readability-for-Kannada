Kannada Readability-Controlled Summarization

Setup:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Training:
Edit CSV_PATH and output paths in train.py
python train.py

Inference:
Edit MODEL_PATH in inference.py
Call summarize(article_text)
