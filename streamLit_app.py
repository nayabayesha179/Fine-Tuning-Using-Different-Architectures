
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the trained model
model_path = "bert_emotion_model"  # folder path (unzipped)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
