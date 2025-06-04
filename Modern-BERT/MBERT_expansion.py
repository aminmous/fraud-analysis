import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from huggingface_hub import HfFolder
from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
import torch._dynamo

torch._dynamo.config.suppress_errors = True

model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

input_seq_length = 4096

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, return_tensors="pt", max_length=input_seq_length)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    # score = f1_score(labels, predictions, labels=labels, pos_label=1, average="binary")
    score = f1_score(labels, predictions, average="macro")

    # return {"f1": float(score) if score == 1 else score}
    return {"f1_macro": float(score)}

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        weight_tensor = self.class_weights.to(logits.device)

        loss_fct = nn.CrossEntropyLoss(weight=weight_tensor)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

def modern_bert_tuner(train_df, val_df, test_df, class_weights, run_name=""):
    print(f"ModernBERT Tuner script started. Run name: {run_name}")

    split_dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df),
        "validation": Dataset.from_pandas(val_df)
    })

    if "label" in split_dataset["train"].features.keys():
        split_dataset = split_dataset.rename_column("label", "labels")  # to match Trainer

    tokenized_dataset = split_dataset.map(tokenize, batched=True, remove_columns=["text"])
    print("Tokenization complete.")

    num_labels = 2
    label2id = {"0": 0, "1": 1}
    id2label = {0: "0", 1: "1"}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        num_labels=num_labels, 
        label2id=label2id, 
        id2label=id2label,
    )

    for name, params in model.named_parameters():
        if not name.startswith("classifier"):
            params.requires_grad = False
    print("Model loaded.")

    training_args = TrainingArguments(
        output_dir= f"expansion-Models/ModernBERT-fraud-expansion-{run_name}",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-5,
        num_train_epochs=15,
        bf16=True, # bfloat16 training 
        optim="adamw_torch_fused", # improved optimizer 
        # logging & evaluation strategies
        logging_strategy="steps",
        logging_steps=100,
        logging_dir=f"./logs/{run_name}",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        # push to hub parameters
        push_to_hub=False,
        hub_strategy="every_save",
        hub_token=HfFolder.get_token(),
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )
    print("Trainer initialized.")
    print("Starting training...")
    trainer.train()
    print("Training complete.")
    print("Evaluating model...")
    preds_out = trainer.predict(tokenized_dataset["test"])

    logits = preds_out.predictions
    preds = np.argmax(logits, axis=1)

    true_labels = preds_out.label_ids

    print("Classification report:")
    print(classification_report(true_labels, preds, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(true_labels, preds))

    prob_pos = logits[:, 1]
    auc = roc_auc_score(true_labels, prob_pos)
    print(f"AUC: {auc:.4f}")

    fpr, tpr, _ = roc_curve(true_labels, prob_pos)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - BERT Fraud Detection")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"expansion-rocs/bert_fraud_roc_{run_name}.png", dpi=300)
