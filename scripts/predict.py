import sys
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.utils.logger import setup_logger
from src.models.ner_model import FinancialNERModel


def format_predictions(predictions, show_all=False):
    """Format predictions for display."""
    output = []
    output.append("\n" + "="*60)
    output.append("Named Entity Recognition Results")
    output.append("="*60)
    
    # Show only entities
    entities_found = False
    for word, label in predictions:
        if label != "O":
            output.append(f"{word:20s} -> {label}")
            entities_found = True
    
    if not entities_found:
        output.append("No entities found.")
    
    # Show all tokens if requested
    if show_all:
        output.append("="*60)
        output.append("\nAll Tokens:")
        output.append("-"*60)
        for word, label in predictions:
            output.append(f"{word:20s} -> {label}")
    
    output.append("="*60)
    
    return "\n".join(output)


def predict_from_text(model, text: str, show_all: bool = False):
    """Predict entities in text."""
    predictions = model.predict(text)
    return format_predictions(predictions, show_all=show_all)


def predict_from_file(model, file_path: Path, show_all: bool = False):
    """Predict entities from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    predictions = model.predict(text)
    return format_predictions(predictions, show_all=show_all)


def interactive_mode(model):
    """Interactive prediction mode."""
    print("\n" + "="*60)
    print("Interactive NER Prediction Mode")
    print("="*60)
    print("Enter text to analyze (or 'quit' to exit)")
    print("-"*60)
    
    while True:
        text = input("\nEnter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not text:
            continue
        
        predictions = model.predict(text)
        print(format_predictions(predictions))


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Named Entity Recognition Prediction")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--file", type=str, help="File containing text to analyze")
    parser.add_argument("--model", type=str, help="Path to saved model (default: models/best_model)")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint folder to load directly")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--show-all", action="store_true", help="Show all tokens including non-entities")
    
    args = parser.parse_args()
    
    # Setup
    config = Config(project_root / "configs" / "config.yaml")
    logger = setup_logger(config.inference.logging_dir, "prediction")
    
    # Determine model path (preference: --checkpoint, --model, default best_model)
    if args.checkpoint:
        model_path = Path(args.checkpoint)
        if not model_path.is_absolute():
            model_path = project_root / model_path
    elif args.model:
        model_path = Path(args.model)
        # If relative path, resolve from project root
        if not model_path.is_absolute():
            model_path = project_root / model_path
    else:
        model_path = project_root / config.paths.model_dir / "best_model"
    
    logger.info(f"Loading model from: {model_path}")
    
    # Initialize and load model (handle missing path/errors)
    model = FinancialNERModel(config)
    try:
        model.load_saved_model(model_path)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        logger.error("If you have training checkpoints, run scripts/save_from_checkpoint.py to convert the latest checkpoint to models/best_model")
        raise
    
    # Prediction modes
    if args.interactive:
        interactive_mode(model)
    elif args.text:
        result = predict_from_text(model, args.text, show_all=args.show_all)
        print(result)
    elif args.file:
        file_path = Path(args.file)
        if not file_path.is_absolute():
            file_path = project_root / file_path
        result = predict_from_file(model, file_path, show_all=args.show_all)
        print(result)
    else:
        # Default: use sample text
        sample_file = project_root / "data" / "sample_text.txt"
        if sample_file.exists():
            logger.info(f"Using sample text from: {sample_file}")
            result = predict_from_file(model, sample_file, show_all=args.show_all)
            print(result)
        else:
            print("No input provided. Use --text, --file, or --interactive")
            parser.print_help()


if __name__ == "__main__":
    main()
