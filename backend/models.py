from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
import evaluate
from seqeval.metrics import classification_report

# Загрузка модели и токенизатора
model_checkpoint = "GanjinZero/PHARM-BERTa"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=3)  # 3 = O, B-ADR, I-ADR

# Загрузка датасета ADE-Corpus-V2
dataset = load_dataset("ade_corpus_v2", "Ade_corpus_v2_classification")

# Преобразование в формат токенов и меток
label2id = {"O": 0, "B-ADR": 1, "I-ADR": 2}
id2label = {v: k for k, v in label2id.items()}

def tokenize_and_align_labels(example):
    tokens = tokenizer(example["text"], truncation=True, return_offsets_mapping=True)
    word_ids = tokens.word_ids()
    labels = [label2id["O"]] * len(tokens["input_ids"])
    if "ADR" in example["text"]:
        labels = [label2id["B-ADR"] if i == 5 else label2id["O"] for i in range(len(labels))]  # демо-логика
    tokens["labels"] = labels
    return tokens

# 4. Токенизация
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=False)

# 5. Метрика
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

#  Параметры обучения
training_args = TrainingArguments(
    output_dir="./pharmberta-ner-ade",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

#  Обучение модели
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()
