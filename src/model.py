from transformers import GPT2Tokenizer, BertTokenizer, RobertaTokenizer, RobertaForSequenceClassification, AlbertTokenizer, AlbertForSequenceClassification
from transformers import GPT2ForSequenceClassification, BertForSequenceClassification, T5Tokenizer, T5ForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import Trainer, TrainingArguments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import torch
import numpy as np
import os

class Model():
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.padding = None
        self.model_dir = None

    def train(self, train_texts, train_labels, val_texts, val_labels):
        if not self.tokenizer or not self.model:
            raise ValueError("Tokenizer and model must be initialized before training.")
        
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=self.padding, max_length=128)
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=self.padding, max_length=128)
        
        self.train_dataset = CustomDataset(train_encodings, train_labels)
        self.val_dataset = CustomDataset(val_encodings, val_labels)

        ## Define training arguments
        # Define training arguments
        training_args = TrainingArguments(
            output_dir='./results',          # Output directory
            num_train_epochs=4,              # Number of epochs
            per_device_train_batch_size=16,  # Batch size for training
            per_device_eval_batch_size=64,   # Batch size for evaluation
            warmup_steps=500,                # Warmup steps for learning rate scheduler
            weight_decay=0.01,               # Strength of weight decay
            logging_dir='./logs',            # Directory for storing logs
            logging_steps=10,
            evaluation_strategy="epoch",      # Evaluate after every epoch
            report_to="none",  # Disable logging to W&B
            save_strategy="epoch"
        )

        ## Initialize Trainer
        # Initialize Trainer
        self.trainer = Trainer(
            model=self.model,                         # Pre-trained BERT model
            args=training_args,                  # Training arguments
            train_dataset=self.train_dataset,         # Training data
            eval_dataset=self.val_dataset,        # Validation data
            compute_metrics=self.compute_metrics

        )

        self.trainer.train()

        # Create a directory to save the model
        # model_dir = "./saved_model"
        model_dir = self.model_dir
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

        print(f"Model and tokenizer saved to {model_dir}")

        # Evaluate the model on the validation set
        results = self.trainer.evaluate()
        print(results)
    
    def predict_from_text(self, text, id2label):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        # tokenizer(train_texts, truncation=True, padding=True, max_length=128)
        outputs = self.model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        class_ = prediction  # Output: 1 (Positive)
        return id2label[class_]
    
    def predict_from_data(self, dataset):
        # Ensure the trainer is initialized
        if not hasattr(self, "trainer") or self.trainer is None:
            if not self.model or not self.tokenizer:
                raise ValueError("Model and tokenizer must be loaded or trained before predicting.")
            # Reinitialize the Trainer
            training_args = TrainingArguments(
                output_dir='./results',          # Output directory
                per_device_eval_batch_size=64,  # Batch size for evaluation
                report_to="none",               # Disable logging to W&B
            )
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                # eval_dataset=dataset,
            )
        preds_output = self.trainer.predict(dataset)
        preds = np.argmax(preds_output.predictions, axis=1)
        labels = preds_output.label_ids

        print(f"Predictions: {preds[:10]}")
        print(f"Labels: {labels[:10]}")
        return preds, labels
    
    def compute_metrics(self, p):
        predictions, labels = p
        if len(predictions.shape)==2:
            preds = np.argmax(predictions, axis=1)
        else:
            preds = predictions

        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        precision = precision_score(labels, preds, average='weighted')
        recall = recall_score(labels, preds, average='weighted')

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def compute_macro_metrics(self, p, label2id):
        predictions, labels = p
        # preds = np.argmax(predictions, axis=1)
        if len(predictions.shape)==2:
            preds = np.argmax(predictions, axis=1)
        else:
            preds = predictions

        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro', labels = list(label2id.values()))
        precision = precision_score(labels, preds, average='macro', labels = list(label2id.values()))
        recall = recall_score(labels, preds, average='macro', labels = list(label2id.values()))

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def _confusion_matrix(self, labels, preds, class_names):
        # Compute confusion matrix
        cm = confusion_matrix(labels, preds)
        print(f"Confusion Matrix: {cm}")  # (6, 6)

        # Plot confusion matrix using Seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                    yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    def load_model(self, model_dir):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

class GPT2(Model):
    def __init__(self, num_labels, model_path, load=False):
        super().__init__()
        self.padding = 'longest'
        if load:
            self.tokenizer, self.model = self.load_model(os.path.join(model_path, "GPT2"))
        else:
            self.tokenizer = self.return_tokenizer()
            self.model = self.create_model(num_labels)

        self.model_dir = os.path.join(model_path, "GPT2")

    def return_tokenizer(self):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def create_model(self, num_labels):
        model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=num_labels)
        model.config.pad_token_id = self.tokenizer.pad_token_id
        return model
    
    def load_model(self, model_path):
        return GPT2Tokenizer.from_pretrained(model_path), GPT2ForSequenceClassification.from_pretrained(model_path)

class BERT(Model):
    def __init__(self, num_labels, model_path, load=False):
        super().__init__()
        self.padding = True
        if load:
            self.tokenizer, self.model = self.load_model(os.path.join(model_path, "BERT"))
        else:
            self.tokenizer = self.return_tokenizer()
            self.model = self.create_model(num_labels)
        self.model_dir = os.path.join(model_path, "BERT")

    def return_tokenizer(self):
        return BertTokenizer.from_pretrained('bert-base-uncased')

    def create_model(self, num_labels):
        return BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    
    def load_model(self, model_path):
        return BertTokenizer.from_pretrained(model_path), BertForSequenceClassification.from_pretrained(model_path)
        
class RoBERTa(Model):
    def __init__(self, num_labels, model_dir, load=False):
        super().__init__()
        self.padding = True
        if load:
            self.tokenizer, self.model = self.load_model(os.path.join(model_dir, "RoBERTa"))
        else:
            self.tokenizer = self.return_tokenizer()
            self.model = self.create_model(num_labels)
        self.model_dir = os.path.join(model_dir, "RoBERTa")

    def return_tokenizer(self):
        return RobertaTokenizer.from_pretrained('roberta-base')
    
    def create_model(self, num_labels):
        return RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)
    
    def load_model(self, model_path):
        return RobertaTokenizer.from_pretrained(model_path), RobertaForSequenceClassification.from_pretrained(model_path)
    
class Albert(Model):
    def _init_(self, num_labels, model_dir):
        super().__init__()
        self.padding = True
        self.tokenizer = self.return_tokenizer()
        self.model = self.create_model(num_labels)
        self.model_dir = os.path.join(model_dir, "ALBERT")
    
    def return_tokenizer(self):
        return AlbertTokenizer.from_pretrained('albert-base-v2')

    def create_model(self, num_labels):
        return AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=num_labels)
    
class T5Base(Model):
    def __init__(self, num_labels, model_dir):
        super().__init__()
        self.padding = True
        self.tokenizer = self.return_tokenizer()
        self.model = self.create_model(num_labels)
        self.model_dir = os.path.join(model_dir, "T5")

    def return_tokenizer(self):
        # Initialize the tokenizer for T5
        return T5Tokenizer.from_pretrained('t5-base')

    def create_model(self, num_labels):
        # Initialize the model for sequence classification
        return T5ForSequenceClassification.from_pretrained('t5-base', num_labels=num_labels)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item