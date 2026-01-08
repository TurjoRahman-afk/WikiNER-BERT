from datasets import load_dataset, Dataset, DatasetDict
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import logging


class DataPreprocessor:
    """Preprocessor for NER datasets."""
    
    def __init__(self, config):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.label2id = None
        self.id2label = None
    
    def load_dataset(self, dataset_name: str = None) -> DatasetDict:
        """
        Load dataset from local files.
        
        Args:
            dataset_name: Name of the dataset to load
        
        Returns:
            DatasetDict containing train, validation, and test splits
        """
        if dataset_name is None:
            dataset_name = self.config.data.dataset_name
        
        self.logger.info(f"Loading dataset: {dataset_name}")
        
        # Try loading from local files first
        data_dir = Path(self.config.paths.data_dir) / "raw" / dataset_name
        if data_dir.exists():
            self.logger.info(f"Loading from local directory: {data_dir}")
            return self.load_from_local_json(data_dir)
        
        # Try loading from Hugging Face
        # But we used the local storage in this project , edit it as you want 
        try:
            dataset = load_dataset(dataset_name)
            self._setup_labels(dataset['train'])
            return dataset
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
    
    def load_from_local_json(self, data_dir: Path) -> DatasetDict:
        """
        Load dataset from local JSON files.
        
        Args:
            data_dir: Directory containing train.json, validation.json, test.json
        
        Returns:
            DatasetDict with splits
        """
        datasets = {}
        
        # Load label mapping
        label_mapping_path = data_dir / "label_mapping.json"
        if label_mapping_path.exists():
            with open(label_mapping_path, 'r') as f:
                label_mapping = json.load(f)
                self.id2label = label_mapping.get('id2label', {})
                self.label2id = label_mapping.get('label2id', {})
                # Convert string keys to int for id2label
                self.id2label = {int(k): v for k, v in self.id2label.items()}
        
        # Load splits
        for split in ['train', 'validation', 'test']:
            file_path = data_dir / f"{split}.json"
            if file_path.exists():
                self.logger.info(f"Loading {split} split from {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                datasets[split] = Dataset.from_list(data)
        
        if not datasets:
            raise ValueError(f"No dataset files found in {data_dir}")
        
        return DatasetDict(datasets)
    
    def _setup_labels(self, dataset: Dataset):
        """Extract label mappings from dataset."""
        if 'ner_tags' in dataset.features:
            label_feature = dataset.features['ner_tags'].feature
            self.label2id = {label: idx for idx, label in enumerate(label_feature.names)}
            self.id2label = {idx: label for label, idx in self.label2id.items()}
    
    def get_labels(self) -> Tuple[Dict, Dict]:
        """
        Get label mappings.
        
        Returns:
            Tuple of (label2id, id2label) dictionaries
        """
        return self.label2id, self.id2label
    
    def save_processed_data(self, dataset: DatasetDict, output_dir: str):
        """
        Save processed dataset to disk.
        
        Args:
            dataset: DatasetDict to save
            output_dir: Directory to save the dataset
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save each split
        for split, data in dataset.items():
            data.save_to_disk(output_path / split)
        
        # Save label mappings
        if self.label2id and self.id2label:
            with open(output_path / "label_mapping.json", 'w') as f:
                json.dump({
                    'label2id': self.label2id,
                    'id2label': self.id2label
                }, f, indent=2)
        
        self.logger.info(f"Saved processed data to {output_path}")
