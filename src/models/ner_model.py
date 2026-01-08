import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging


class FinancialNERModel:
    #BERT-based model for Named Entity Recognition.
    
    def __init__(self, config, label2id: Dict = None, id2label: Dict = None):
        """
        Initialize the NER model.
        
        Args:
            config: Configuration object
            label2id: Label to ID mapping
            id2label: ID to label mapping
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup labels
        self.label2id = label2id
        self.id2label = id2label
        """
        label2id = {"O": 0, "B-PER": 1, "I-PER": 2}
        id2label = {0: "O", 1: "B-PER", 2: "I-PER"}
        """
        if self.label2id:
            self.num_labels = len(self.label2id)
        else:
            self.num_labels = config.model.num_labels
        
        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
    
    def setup_labels(self, label2id: Dict, id2label: Dict):
        """Setup label mappings."""
        self.label2id = label2id # saves label2id 
        self.id2label = id2label # saves id2label
        self.num_labels = len(label2id) # counts the numbber of labels in label2id
        self.config.update('model.num_labels', self.num_labels) # update the config object with the new num_labels
    
    def load_model(self):
        """Load pre-trained model and tokenizer."""
        # after this the model is ready to train and predict 
        self.logger.info(f"Loading model: {self.config.model.name}") # loads bert pretrained model
        
        self.tokenizer = AutoTokenizer.from_pretrained( # tokenize the texts 
            self.config.model.name,
            add_prefix_space=True
        )
        
        self.model = AutoModelForTokenClassification.from_pretrained(
            #loads a pretrained model for token classification tasks
            self.config.model.name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        self.model.to(self.device)
        self.logger.info(f"Model loaded on device: {self.device}")
    
    def tokenize_and_align_labels(self, examples):
        """
        Tokenize inputs and align labels with tokens.
        
        Args:
            examples: Batch of examples from dataset
        
        Returns:
            Tokenized inputs with aligned labels
            #Tokens:   ["John", "lives", "in", "New", "York"]
            #Labels:   [ B-PER,    O,      O,   B-LOC, I-LOC ]
        """
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=self.config.model.max_length,
            padding="max_length"
        )
        
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def train(self, train_dataset: Dataset, eval_dataset: Dataset = None):
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
        """
        if self.model is None:
            self.load_model()
        
        # Tokenize datasets
        self.logger.info("Tokenizing datasets...")
        train_tokenized = train_dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        eval_tokenized = None
        if eval_dataset:
            eval_tokenized = eval_dataset.map(
                self.tokenize_and_align_labels,
                batched=True,
                remove_columns=eval_dataset.column_names
            )
        
        # Setup training arguments
        # Determine evaluation strategy key compatible with installed transformers
        eval_strategy = None
        try:
            eval_strategy = self.config.get('training.evaluation_strategy')
        except Exception:
            eval_strategy = self.config.get('training.eval_strategy')

        training_args = TrainingArguments(
            output_dir=self.config.get('training.output_dir'),
            num_train_epochs=self.config.get('training.num_train_epochs'),
            per_device_train_batch_size=self.config.get('training.per_device_train_batch_size'),
            per_device_eval_batch_size=self.config.get('training.per_device_eval_batch_size'),
            warmup_steps=self.config.get('training.warmup_steps'),
            weight_decay=self.config.get('training.weight_decay'),
            logging_dir=self.config.get('training.logging_dir'),
            logging_steps=self.config.get('training.logging_steps'),
            eval_strategy=eval_strategy,
            save_strategy=self.config.get('training.save_strategy'),
            load_best_model_at_end=self.config.get('training.load_best_model_at_end'),
            metric_for_best_model=self.config.get('training.metric_for_best_model'),
            greater_is_better=self.config.get('training.greater_is_better'),
            save_total_limit=self.config.get('training.save_total_limit'),
            learning_rate=self.config.get('training.learning_rate'),
        )
        
        # Create callbacks
        callbacks = []
        if hasattr(self.config.training, 'early_stopping_patience'):
            early_stopping_patience = self.config.training.early_stopping_patience
            if early_stopping_patience and early_stopping_patience > 0:
                callbacks.append(
                    EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
                )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks if callbacks else None
        )
        
        # Train
        self.logger.info("Starting training...")
        train_result = trainer.train()
        
        # Save model
        self.logger.info("Training completed. Saving model...")
        self.save_model(Path(self.config.paths.model_dir) / "best_model")
        
        return train_result
    
    def predict(self, text: str) -> List[Tuple[str, str]]:
        """
        Predict NER tags for input text.
        
        Args:
            text: Input text
        
        Returns:
            List of (word, label) tuples
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize
        words = text.split()
        tokenizer_output = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.model.max_length,
            padding="max_length"
        )
        
        # Get word_ids before moving to device
        word_ids = tokenizer_output.word_ids(batch_index=0)
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in tokenizer_output.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Align predictions with words
        predicted_labels = []
        
        for position, word_idx in enumerate(word_ids):
            if word_idx is not None:
                # Only take the first subword prediction for each word
                if position == 0 or word_ids[position - 1] != word_idx:
                    label_id = predictions[0][position].item()
                    label = self.id2label.get(label_id, "O")
                    predicted_labels.append(label)
        
        # Match words with labels
        result = list(zip(words, predicted_labels))
        return result
    
    def compute_metrics(self, pred):
        """
        Compute evaluation metrics.
        
        Args:
            pred: Predictions from trainer
        
        Returns:
            Dictionary of metrics
        """
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Flatten lists
        flat_predictions = [item for sublist in true_predictions for item in sublist]
        flat_labels = [item for sublist in true_labels for item in sublist]
        
        # Calculate metrics
        accuracy = accuracy_score(flat_labels, flat_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            flat_labels, flat_predictions, average='weighted', zero_division=0
        )
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def save_model(self, save_path):
        """
        Save model and tokenizer.
        
        Args:
            save_path: Path or string to save the model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer (transformers accepts str or Path)
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        
        # Save label mappings
        with open(save_path / "label_mapping.json", 'w', encoding='utf-8') as f:
            json.dump({
                'label2id': self.label2id,
                'id2label': self.id2label
            }, f, indent=2)
        
        self.logger.info(f"Model saved to {save_path}")
    
    def load_saved_model(self, model_path):
        """
        Load a saved model.
        
        Args:
            model_path: Path or string to the saved model
        """
        model_path = Path(model_path)
        self.logger.info(f"Loading model from {model_path}")
        
        # Load label mappings (check model path first, then dataset)
        label_map_file = model_path / "label_mapping.json"
        if not label_map_file.exists():
            # Fallback: look in dataset folder
            dataset_name = self.config.data.dataset_name
            project_root = Path.cwd()
            dataset_label_map = project_root / "data" / "raw" / dataset_name / "label_mapping.json"
            if dataset_label_map.exists():
                label_map_file = dataset_label_map
                self.logger.info(f"Using label mapping from dataset: {label_map_file}")
            else:
                raise FileNotFoundError(f"label_mapping.json not found in {model_path} or {dataset_label_map}")
        
        with open(label_map_file, 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
            self.label2id = label_mapping['label2id']
            self.id2label = {int(k): v for k, v in label_mapping['id2label'].items()}
            self.num_labels = len(self.label2id)
        
        # Check if path exists to use local_files_only
        local_only = model_path.exists()
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                local_files_only=local_only
            )
            self.model = AutoModelForTokenClassification.from_pretrained(
                str(model_path),
                local_files_only=local_only
            )
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
        
        self.model.to(self.device)
        self.logger.info(f"Model loaded on device: {self.device}")
