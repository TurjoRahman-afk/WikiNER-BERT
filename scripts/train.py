"""
Training script for Financial NER model
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.ner_model import FinancialNERModel
from src.data.preprocessor import DataPreprocessor
from src.utils.config import Config
from src.utils.logger import setup_logger
import json
import time


def main():
    """Main training function"""
    
    # Load configuration
    config = Config(str(project_root / "configs" / "config.yaml"))

    # Setup logger
    logger = setup_logger(
        name="training",
        log_dir=config.get('paths.logs_dir', './logs')
    )
    
    logger.info("="*60)
    logger.info("Starting Financial NER Training")
    logger.info("="*60)
    
    # Step 1: Load and preprocess data
    logger.info("\n[Step 1] Loading dataset...")
    preprocessor = DataPreprocessor(config)

    # Prefer dataset folder named in config under data/raw/
    dataset_name = config.get('data.dataset_name', 'wikiner')
    local_data_dir = project_root / 'data' / 'raw'
    preferred_path = local_data_dir / dataset_name

    dataset_loaded = False
    if preferred_path.exists() and (preferred_path / 'train.json').exists():
        logger.info(f"Loading local dataset from: {preferred_path}")
        dataset = preprocessor.load_from_local_json(preferred_path)
        dataset_loaded = True
    else:
        # Fallback: scan for first dataset folder containing train.json
        if local_data_dir.exists():
            for d in local_data_dir.iterdir():
                if d.is_dir() and (d / 'train.json').exists():
                    logger.info(f"Found local dataset at: {d}")
                    dataset = preprocessor.load_from_local_json(d)
                    dataset_loaded = True
                    break

    if not dataset_loaded:
        logger.info("No local dataset found. Attempting to load from HuggingFace...")
        try:
            dataset = preprocessor.load_dataset(dataset_name)
            dataset_loaded = True
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    # dataset is a DatasetDict
    train_dataset = dataset.get('train')
    val_dataset = dataset.get('validation') or dataset.get('valid')
    test_dataset = dataset.get('test')
    
    # Step 2: Initialize model
    logger.info("\n[Step 2] Initializing model...")
    label2id, id2label = preprocessor.get_labels()

    model = FinancialNERModel(config, label2id=label2id, id2label=id2label)
    model.load_model()
    
    # Step 3: Tokenize datasets
    logger.info("\n[Step 3] Tokenizing datasets...")
    train_dataset = train_dataset.map(
        model.tokenize_and_align_labels,
        batched=True,
        desc="Tokenizing train dataset"
    )
    val_dataset = val_dataset.map(
        model.tokenize_and_align_labels,
        batched=True,
        desc="Tokenizing validation dataset"
    )
    test_dataset = test_dataset.map(
        model.tokenize_and_align_labels,
        batched=True,
        desc="Tokenizing test dataset"
    )
    
    logger.info("Tokenization completed!")
    
    # Step 4: Train model
    logger.info("\n[Step 4] Training model...")
    logger.info(f"Training configuration:")
    logger.info(f"  Learning rate: {config.get('training.learning_rate')}")
    logger.info(f"  Batch size: {config.get('training.per_device_train_batch_size')}")
    logger.info(f"  Epochs: {config.get('training.num_train_epochs')}")
    logger.info(f"  Output dir: {config.get('training.output_dir')}")
    
    # Step 4: Train model using FinancialNERModel.train()
    logger.info("\n[Step 4] Training model...")
    start = time.time()
    train_result = model.train(train_dataset, eval_dataset=val_dataset)
    duration = time.time() - start
    logger.info(f"Training finished in {duration:.2f} seconds")
    
    # Step 5: Evaluate on test set
    logger.info("\n[Step 5] Evaluating on test set...")
    # Evaluate on test dataset
    from transformers import Trainer, TrainingArguments

    test_tokenized = test_dataset.map(
        model.tokenize_and_align_labels,
        batched=True,
        remove_columns=test_dataset.column_names
    )

    eval_args = TrainingArguments(
        output_dir=config.get('training.output_dir', './models/checkpoints'), # save model check points here
        per_device_eval_batch_size=config.get('training.per_device_eval_batch_size', 16), # eval batch size =16
    )

    trainer = Trainer(
        model=model.model,
        args=eval_args, # evaluation arguments
        compute_metrics=model.compute_metrics,
    )

    results = trainer.evaluate(test_tokenized)
    
    logger.info("\nTest Set Results:")
    logger.info(f"  Accuracy: {results['eval_accuracy']:.4f}")
    logger.info(f"  Precision: {results['eval_precision']:.4f}")
    logger.info(f"  Recall: {results['eval_recall']:.4f}")
    logger.info(f"  F1-Score: {results['eval_f1']:.4f}")
    
    # Step 6: Save model
    logger.info("\n[Step 6] Saving model...")
    save_dir = config.get('paths.model_save_dir', './models/best_model')
    model.save_model(save_dir)
    
    # Save results
    results_file = Path(config.get('paths.results_dir', './results')) / 'training_results.json'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    
    # Step 7: Test predictions
    # not fully done but you can try with only 3 sentences and add more later 
    logger.info("\n[Step 7] Testing predictions...")
    test_sentences = [
        "Apple Inc reported revenue of 394 billion dollars in fiscal year 2023",
        "The company's market cap reached 3 trillion dollars last quarter",
        "EBITDA increased by 15 percent compared to the previous year"
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        logger.info(f"\nTest Sentence {i}: {sentence}")
        predictions = model.predict(sentence)
        
        logger.info("Predictions:")
        for token, label in predictions:
            if label != 'O':  # Only show entity predictions
                logger.info(f"  {token:15} -> {label}")
    
    logger.info("\n" + "="*60)
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {save_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
