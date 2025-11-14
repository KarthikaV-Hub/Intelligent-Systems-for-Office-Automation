pip install transformers datasets sentencepiece accelerate

!pip install evaluate

# IMPORTS
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
import evaluate

# 1. LOAD DATASET (IMDb dataset)
dataset_obj = load_dataset("imdb")

# 2. LOAD TOKENIZER
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# 3. TOKENIZATION FUNCTION
def encode_samples(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

# Apply tokenization
tokenized_set = dataset_obj.map(encode_samples, batched=True)

# Rename column to "labels"
tokenized_set = tokenized_set.rename_column("label", "labels")

# Remove unnecessary columns
tokenized_set = tokenized_set.remove_columns(["text"])

# Set PyTorch format
tokenized_set.set_format("torch")

# Train & Test splits
train_portion = tokenized_set["train"]
test_portion = tokenized_set["test"]

# 4. LOAD PRETRAINED DistilBERT FOR CLASSIFICATION
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# 5. METRICS (Accuracy)
accuracy_metric = evaluate.load("accuracy")

!pip install -U transformers

# 6. TRAINING ARGUMENTS
train_cfg = TrainingArguments(
    output_dir="./distilbert_sentiment_model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01
)


# 7. TRAINER OBJECT
trainer = Trainer(
    model=model,
    args=train_cfg,
    train_dataset=train_portion,
    eval_dataset=test_portion,
    tokenizer=tokenizer,
    compute_metrics=compute_scores
)

import os
os.environ["WANDB_DISABLED"] = "true"

# 8. TRAINING
trainer.train()

import torch

def predict_sentiment(text):
    model.eval()

    device = next(model.parameters()).device   # get model device (cpu or cuda)

    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)   # <-- move tokens to same device

    with torch.no_grad():
        outputs = model(**tokens)

    # Get predicted label
    predicted = torch.argmax(outputs.logits, dim=1).item()

    return "Positive 😀" if predicted == 1 else "Negative 😡"

print(predict_sentiment("The movie was absolutely fantastic!"))
print(predict_sentiment("Worst film I have ever watched."))


