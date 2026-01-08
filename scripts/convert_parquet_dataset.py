"""
Script to convert WikiNER parquet files to project JSON format
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import json
from typing import Dict, List


def convert_parquet_to_json(parquet_file: Path, output_file: Path):

    print(f"Reading {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Convert to project format
    data = []
    for idx, row in df.iterrows():
        
        tokens = row['words'] if 'words' in row else row['tokens']
        ner_tags = row['ner_tags']
        
        # Convert numpy arrays to lists if needed
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        if hasattr(ner_tags, 'tolist'):
            ner_tags = ner_tags.tolist()
        
        data.append({
            'id': str(idx),
            'tokens': tokens,
            'ner_tags': ner_tags
        })
    
    # Save to JSON
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved {len(data)} sentences")
    return len(data)


def create_label_mapping(train_data: List[Dict]) -> tuple:
    """
    Create label mappings from training data.
    
    Args:
        train_data: Training data
    
    Returns:
        Tuple of (label2id, id2label)
    """
    # Collect all unique labels
    all_labels = set()
    for example in train_data:
        all_labels.update(example['ner_tags'])
    
    # Sort labels (ensure O is first)
    labels = sorted(list(all_labels))
    if 0 in labels:
        labels.remove(0)
        labels = [0] + labels
    
    # Create mappings
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    return label2id, id2label


def main():
    """Main conversion function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert WikiNER parquet to JSON")
    parser.add_argument("--train", type=str, required=True, help="Path to train.parquet")
    parser.add_argument("--test", type=str, required=True, help="Path to test.parquet")
    parser.add_argument("--output_dir", type=str, default="data/raw/wikiner", 
                      help="Output directory")
    parser.add_argument("--val_split", type=float, default=0.1, 
                      help="Validation split ratio (from train)")
    
    args = parser.parse_args()
    
    # Setup paths
    train_parquet = Path(args.train)
    test_parquet = Path(args.test)
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("WikiNER Parquet to JSON Converter")
    print("="*60)
    
    # Convert training data
    print("\n1. Converting training data...")
    train_json = output_dir / "train_full.json"
    num_train = convert_parquet_to_json(train_parquet, train_json)
    
    # Load train data for splitting
    with open(train_json, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # Split into train and validation
    val_size = int(len(train_data) * args.val_split)
    val_data = train_data[:val_size]
    train_data = train_data[val_size:]
    
    # Save split train
    train_output = output_dir / "train.json"
    print(f"\nSaving training split: {len(train_data)} sentences...")
    with open(train_output, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2)
    
    # Save validation
    val_output = output_dir / "validation.json"
    print(f"Saving validation split: {len(val_data)} sentences...")
    with open(val_output, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2)
    
    # Remove temporary file
    train_json.unlink()
    
    # Convert test data
    print("\n2. Converting test data...")
    test_output = output_dir / "test.json"
    num_test = convert_parquet_to_json(test_parquet, test_output)
    
    # Create label mappings from training data
    print("\n3. Creating label mappings...")
    
    # WikiNER standard labels
    # 0: O (Outside), 1: PER (Person), 2: ORG (Organization), 
    # 3: LOC (Location), 4: MISC (Miscellaneous)
    id2label = {
        0: "O",
        1: "LOC",
        2: "PER",
        3: "MISC",# theres nothing labeled 3 in the dataset so we skip this and named it MISC
        4: "ORG"
    }
    label2id = {v: k for k, v in id2label.items()}
    
    # Save label mappings
    label_mapping_file = output_dir / "label_mapping.json"
    with open(label_mapping_file, 'w', encoding='utf-8') as f:
        json.dump({
            'label2id': label2id,
            'id2label': id2label
        }, f, indent=2)
    
    print(f"Label mappings saved to {label_mapping_file}")
    print(f"\nLabels: {list(label2id.keys())}")
    
    # Summary
    print("\n" + "="*60)
    print("Conversion Summary")
    print("="*60)
    print(f"Training sentences:   {len(train_data):,}")
    print(f"Validation sentences: {len(val_data):,}")
    print(f"Test sentences:       {num_test:,}")
    print(f"Total sentences:      {len(train_data) + len(val_data) + num_test:,}")
    print(f"\nOutput directory: {output_dir}")
    print("="*60)
    print("\nFiles created:")
    print(f"  - {train_output.relative_to(project_root)}")
    print(f"  - {val_output.relative_to(project_root)}")
    print(f"  - {test_output.relative_to(project_root)}")
    print(f"  - {label_mapping_file.relative_to(project_root)}")
    print("\nDataset is ready for training!")


if __name__ == "__main__":
    main()
