# Named Entity Recognition (NER) with BERT

A production-ready NER system using BERT fine-tuned on WikiNER dataset to identify and classify named entities (Person, Location, Organization) in text. Built with a modular architecture for easy customization and deployment.

## 🌟 Features

- **Pre-trained BERT Model**: Leverages `bert-base-uncased` fine-tuned on WikiNER dataset
- **5 Entity Types**: Person (PER), Location (LOC), Organization (ORG), Miscellaneous (MISC), and Outside (O)
- **High Accuracy**: Achieved 98.77% F1-score on test set
- **Modular Architecture**: Clean separation of concerns with reusable components
- **Easy Configuration**: YAML-based configuration management
- **Multiple Prediction Modes**: CLI, interactive, and batch processing
- **Early Stopping**: Configurable patience for optimal training
- **Comprehensive Logging**: Track training and inference progress
- **Production Ready**: Structured for deployment and scalability

## 📁 Project Structure

```
BERT NER model/
├── src/
│   ├── models/
│   │   └── ner_model.py          # BERT-based NER model
│   ├── data/
│   │   └── preprocessor.py       # Data loading and preprocessing
│   └── utils/
│       ├── config.py              # Configuration management
│       └── logger.py              # Logging utilities
├── scripts/
│   ├── train.py                   # Training script
│   ├── predict.py                 # Prediction/inference script
│   ├── convert_parquet_dataset.py # Parquet to JSON converter
│   └── convert_csv_dataset.py     # CSV to JSON converter
├── configs/
│   └── config.yaml                # Configuration file
├── data/
│   ├── raw/
│   │   └── wikiner/               # WikiNER dataset (116k+ train sentences)
│   │       ├── train.json
│   │       ├── validation.json
│   │       ├── test.json
│   │       └── label_mapping.json
│   ├── processed/                 # Preprocessed data
│   └── cache/                     # HuggingFace cache
├── models/
│   ├── best_model/                # Saved trained model (F1: 0.9877)
│   └── checkpoints/               # Training checkpoints
├── results/
│   └── training_results.json      # Training metrics
├── logs/                          # Execution logs
├── requirements.txt               # Python dependencies
└── PROJECT_README.md              # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Navigate to project directory
cd "BERT NER model"

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset

The model is trained on **WikiNER dataset** with the following statistics:
- **Training Set**: 116,438 sentences
- **Validation Set**: 12,938 sentences
- **Test Set**: 14,398 sentences
- **Total Tokens**: 3,488,498
- **Entity Tokens**: 356,190 (10.2%)
- **Labels**: 5 (O, LOC, PER, MISC, ORG)

### 3. Train the Model

```bash
# Train with configuration from configs/config.yaml
python scripts/train.py

# The model will:
# - Load WikiNER dataset from data/raw/wikiner/
# - Preprocess and tokenize the data
# - Fine-tune BERT on named entities
# - Save the trained model to ./models/best_model/
# - Save metrics to ./results/training_results.json
```

**Training Configuration** (in `configs/config.yaml`):
- **Model**: `bert-base-uncased`
- **Learning Rate**: `2e-5`
- **Batch Size**: `32` (per device)
- **Epochs**: `3`
- **Early Stopping Patience**: `2`
- **Max Sequence Length**: `128`
- **Warmup Steps**: `1000`
- **Weight Decay**: `0.01`

### 4. Training Results

Our model achieved excellent performance on the WikiNER test set:

| Metric | Score |
|--------|-------|
| **Accuracy** | 98.77% |
| **Precision** | 98.78% |
| **Recall** | 98.77% |
| **F1-Score** | 98.77% |
| **Eval Loss** | 0.0369 |

**Training Details**:
- Total Training Steps: ~10,917 (3 epochs × 3,639 steps/epoch)
- Training Time: ~1 hour 5 minutes
- Hardware: NVIDIA GeForce RTX 5060 Laptop GPU
- Evaluation Speed: 230.6 samples/second

### 5. Make Predictions

#### Option A: Interactive Mode
```bash
python scripts/predict.py --interactive

# Enter text to analyze in real-time
> Mark is the founder of Apple Corporation in California
```

#### Option B: Analyze Specific Text
```bash
python scripts/predict.py --text "Mark is the founder of Apple Corporation in California"

# Output:
# Mark         -> PER
# is           -> O
# the          -> O
# founder      -> O
# of           -> O
# Apple        -> ORG
# Corporation  -> ORG
# in           -> O
# California   -> LOC
```

#### Option C: Show All Tokens (including non-entities)
```bash
python scripts/predict.py --text "Your text here" --show-all
```

#### Option D: Batch Process a File
```bash
python scripts/predict.py --file input.txt
```

#### Option E: Run Example Predictions
```bash
python scripts/predict.py
```

#### Option F: Use Specific Checkpoint
```bash
python scripts/predict.py --model models/checkpoints/checkpoint-10917 --text "Your text here"
```
## 🎯 Entity Types

The model recognizes 5 entity types:

| Label | Description | Example |
|-------|-------------|---------|
| **O** | Outside (non-entity) | "the", "is", "of" |
| **PER** | Person names | "Mark Zuckerberg", "John Smith" |
| **LOC** | Locations | "California", "New York", "London" |
| **ORG** | Organizations | "Apple Corporation", "Microsoft", "United Nations" |
| **MISC** | Miscellaneous entities | Other named entities not fitting above categories |

## 📊 Model Architecture

**Base Model**: `bert-base-uncased`
- **Architecture**: Transformer encoder with 12 layers
- **Hidden Size**: 768
- **Attention Heads**: 12
- **Parameters**: ~110M
- **Vocabulary Size**: 30,522 tokens
- **Max Sequence Length**: 128 tokens (configurable)

**Fine-tuning Approach**:
- Added token classification head on top of BERT
- Trained end-to-end with label alignment for subword tokens
- Early stopping based on validation F1 score
- Best model selection using F1 as metric

## 🎯 Usage Examples

### Python API - Training Custom Model

```python
from src.models.ner_model import FinancialNERModel
from src.data.preprocessor import DataPreprocessor

# Load data from local directory
preprocessor = DataPreprocessor("wikiner")
dataset = preprocessor.load_from_local_json("data/raw/wikiner")
train_ds, val_ds, test_ds = preprocessor.convert_to_model_format()

# Initialize and train model
model = FinancialNERModel(model_name="bert-base-uncased")
model.setup_labels(preprocessor.get_all_labels())
model.load_model()

# Tokenize data
train_ds = train_ds.map(model.tokenize_and_align_labels, batched=True)
val_ds = val_ds.map(model.tokenize_and_align_labels, batched=True)

# Train
trainer = model.train(
    train_dataset=train_ds,
    val_dataset=val_ds,
    output_dir="./models/checkpoints",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    num_train_epochs=3,
    early_stopping_patience=2
)

# Save
model.save_model("./models/my_model")
```
### Python API - Making Predictions

```python
from src.models.ner_model import FinancialNERModel

# Load trained model
model = FinancialNERModel()
model.load_saved_model("./models/best_model")

# Predict on text
text = "Mark is the founder of Apple Corporation in California"
predictions = model.predict(text)

# Display results
for token, label in predictions:
    print(f"{token:15} -> {label}")

# Output:
# Mark            -> PER
# is              -> O
# the             -> O
# founder         -> O
# of              -> O
# Apple           -> ORG
# Corporation     -> ORG
# in              -> O
# California      -> LOC
```

## ⚙️ Configuration

Edit `configs/config.yaml` to customize training and model settings:

```yaml
model:
  name: "bert-base-uncased"
  max_length: 128
  num_labels: null  # Auto-detected from dataset

training:
  output_dir: "./models/checkpoints"
  learning_rate: 2.0e-5
  per_device_train_batch_size: 32
  per_device_eval_batch_size: 16
  num_train_epochs: 3
  weight_decay: 0.01
  warmup_steps: 1000
  logging_steps: 500
  eval_strategy: "epoch"
  save_strategy: "epoch"
  load_best_model_at_end: true
  save_total_limit: 2
  early_stopping_patience: 2  # Stop if no improvement for 2 epochs
  seed: 42

data:
  dataset_name: "wikiner"
  processed_data_dir: "./data/processed"
  raw_data_dir: "./data/raw"
  cache_dir: "./data/cache"

paths:
  model_save_dir: "./models/best_model"
  results_dir: "./results"
  logs_dir: "./logs"
```

## 🔧 Advanced Usage

### Using Different Base Models

```python
# RoBERTa
model = FinancialNERModel(model_name="roberta-base")

# DistilBERT (faster, smaller)
model = FinancialNERModel(model_name="distilbert-base-uncased")

# Custom BERT variant
model = FinancialNERModel(model_name="bert-large-cased")
```

### Custom Dataset Integration

Convert your dataset to the required JSON format:

```json
[
  {
    "id": "0",
    "tokens": ["Apple", "Inc", "is", "in", "California"],
    "ner_tags": [2, 2, 0, 0, 1]
  }
]
```

With `label_mapping.json`:
```json
{
  "id2label": {
    "0": "O",
    "1": "LOC",
    "2": "ORG",
    "3": "MISC",
    "4": "PER"
  },
  "label2id": {
    "O": 0,
    "LOC": 1,
    "ORG": 2,
    "MISC": 3,
    "PER": 4
  }
}
```

Use converter scripts:
```bash
# Convert CSV to project format
python scripts/convert_csv_dataset.py

# Convert Parquet to project format
python scripts/convert_parquet_dataset.py
```

## 🛠️ Development

### System Requirements

- **Python**: 3.8+ (tested on 3.11)
- **PyTorch**: 2.0+ with CUDA 12.8 support
- **RAM**: 8GB+ (16GB+ recommended for training)
- **GPU**: CUDA-capable GPU recommended (tested on RTX 5060)
- **Storage**: 5GB+ for models and datasets

### Key Dependencies

```
torch==2.9.1+cu128
transformers>=4.30.0
datasets>=2.14.0
scikit-learn>=1.3.0
pyyaml>=6.0
numpy>=1.24.0
```

## 📄 License

This project is open source. Component licenses:
- **BERT Model**: Apache 2.0 License (Google)
- **HuggingFace Transformers**: Apache 2.0 License
- **WikiNER Dataset**: Creative Commons Attribution 3.0 Unported License

Free for commercial and personal use with attribution.

## 🎓 Citation

If you use this project in your research, please cite:

```bibtex
@misc{wikiner-ner-bert,
  title={Named Entity Recognition with BERT on WikiNER},
  author={Your Name},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/bert-ner-model}}
}
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Support for additional entity types
- Multi-language support
- Model distillation for faster inference
- API deployment examples
- Extended documentation

## 📧 Contact & Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Submit a pull request
- Contact: [us.khan.2002@gmail.com]

## 🙏 Acknowledgments

- **Google AI** for BERT architecture
- **HuggingFace** for Transformers library
- **WikiNER** dataset contributors
- **PyTorch** team for the deep learning framework

---

**Project Status**: ✅ Production Ready  
**Last Updated**: January 2026  
**Version**: 1.0.0

Contributions welcome! Areas for improvement:
- Additional evaluation metrics
- Support for more base models
- API/web interface
- Docker deployment
- Real-time inference optimization

## 📧 Contact

For questions or issues, please open a GitHub issue.

## 🙏 Acknowledgments

- **FiNER-139 Dataset**: [nlpaueb/finer-139](https://huggingface.co/datasets/nlpaueb/finer-139)
- **HuggingFace**: For the Transformers library and model hub
- **BERT**: Google's groundbreaking NLP model

## 📚 References

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [FiNER-139 Dataset](https://huggingface.co/datasets/nlpaueb/finer-139)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

---

**Built with ❤️ for Financial NLP**
