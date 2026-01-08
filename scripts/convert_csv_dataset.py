"""
Convert CSV NER dataset to training format
"""
import sys
from pathlib import Path
import json
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def convert_csv_to_json(csv_path: str, output_name: str = "my-dataset"):
    """
    Convert CoNLL-format CSV to JSON training format
    
    Args:
        csv_path: Path to CSV file
        output_name: Name for output directory
    """
    print(f"Converting CSV: {csv_path}")
    print("="*60)
    
    try:
        # Read CSV
        print("\n[1] Reading CSV file...")
        df = pd.read_csv(csv_path, encoding='latin1')
        
        print(f"   Total rows: {len(df):,}")
        print(f"   Columns: {list(df.columns)}")
        
        # Determine column names
        sentence_col = None
        word_col = None
        tag_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'sentence' in col_lower:
                sentence_col = col
            elif 'word' in col_lower or 'token' in col_lower:
                word_col = col
            elif 'tag' in col_lower and 'pos' not in col_lower:
                tag_col = col
        
        print(f"   Sentence column: {sentence_col}")
        print(f"   Word column: {word_col}")
        print(f"   Tag column: {tag_col}")
        
        # Group by sentences
        print("\n[2] Grouping by sentences...")
        sentences = []
        current_sentence = []
        current_tags = []
        
        for idx, row in df.iterrows():
            # Check if new sentence
            if pd.notna(row[sentence_col]) and current_sentence:
                # Save previous sentence
                sentences.append({
                    'tokens': current_sentence,
                    'tags': current_tags
                })
                current_sentence = []
                current_tags = []
            
            # Add word and tag
            word = str(row[word_col])
            tag = str(row[tag_col])
            
            if word and word != 'nan':
                current_sentence.append(word)
                current_tags.append(tag)
        
        # Add last sentence
        if current_sentence:
            sentences.append({
                'tokens': current_sentence,
                'tags': current_tags
            })
        
        print(f"   Total sentences: {len(sentences):,}")
        
        # Get unique tags and create label mapping
        print("\n[3] Creating label mapping...")
        all_tags = set()
        for sent in sentences:
            all_tags.update(sent['tags'])
        
        all_tags = sorted(list(all_tags))
        label_mapping = {
            'id2label': {i: tag for i, tag in enumerate(all_tags)},
            'label2id': {tag: i for i, tag in enumerate(all_tags)}
        }
        
        print(f"   Unique tags: {len(all_tags)}")
        print(f"   Tags: {all_tags}")
        
        # Convert tags to IDs
        print("\n[4] Converting tags to IDs...")
        for sent in sentences:
            sent['ner_tags'] = [label_mapping['label2id'][tag] for tag in sent['tags']]
            del sent['tags']
        
        # Split into train/validation/test (80/10/10)
        print("\n[5] Splitting dataset...")
        total = len(sentences)
        train_size = int(0.8 * total)
        val_size = int(0.1 * total)
        
        train_data = sentences[:train_size]
        val_data = sentences[train_size:train_size + val_size]
        test_data = sentences[train_size + val_size:]
        
        print(f"   Train: {len(train_data):,} sentences")
        print(f"   Validation: {len(val_data):,} sentences")
        print(f"   Test: {len(test_data):,} sentences")
        
        # Save to output directory
        print("\n[6] Saving to JSON files...")
        output_dir = project_root / "data" / "raw" / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        splits = {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
        
        for split_name, split_data in splits.items():
            # Add IDs
            for idx, item in enumerate(split_data):
                item['id'] = str(idx)
            
            output_file = output_dir / f"{split_name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2)
            
            print(f"   ✅ {split_name}.json saved ({len(split_data)} examples)")
        
        # Save label mapping
        mapping_file = output_dir / "label_mapping.json"
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(label_mapping, f, indent=2)
        
        print(f"   ✅ Label mapping saved")
        
        print(f"\n{'='*60}")
        print(f"✅ SUCCESS! Dataset converted and ready for training")
        print(f"📁 Location: {output_dir}")
        print(f"\nThe dataset will be auto-detected by train.py")
        
        # Show sample
        print(f"\n{'='*60}")
        print("Sample from training data:")
        sample = train_data[0]
        print(f"Tokens: {sample['tokens'][:10]}")
        print(f"Tags: {[label_mapping['id2label'][t] for t in sample['ner_tags'][:10]]}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error converting dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Default path
    csv_path = r"D:\DESKTOP\Projects\NER dataset.csv"
    output_name = "ner-dataset"
    
    # Allow custom path from command line
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_name = sys.argv[2]
    
    print(f"CSV Path: {csv_path}")
    print(f"Output Name: {output_name}\n")
    
    convert_csv_to_json(csv_path, output_name)
