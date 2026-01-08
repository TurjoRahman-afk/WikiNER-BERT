import argparse
from pathlib import Path
import json
import sys

# Ensure project root is on sys.path so `src` imports work
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.models.ner_model import FinancialNERModel
from transformers import AutoTokenizer, AutoModelForTokenClassification


def find_label_mapping(dataset_path: Path, checkpoint_path: Path):
    # Prefer dataset mapping
    ds_map = dataset_path / 'label_mapping.json'
    if ds_map.exists():
        return ds_map
    ck_map = checkpoint_path / 'label_mapping.json'
    if ck_map.exists():
        return ck_map
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path (folder)')
    parser.add_argument('--dataset', type=str, default='data/raw/wikiner', help='Path to dataset folder with label_mapping.json')
    parser.add_argument('--out', type=str, default='models/best_model', help='Output model directory')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    cfg = Config(project_root / 'configs' / 'config.yaml')

    # Resolve checkpoint
    ck = Path(args.checkpoint) if args.checkpoint else project_root / 'models' / 'checkpoints'
    if ck.exists() and ck.is_dir():
        # If a directory of checkpoints provided, pick the latest by name
        if any(ck.iterdir()):
            # pick highest numeric suffix if names like checkpoint-####
            candidates = [d for d in ck.iterdir() if d.is_dir()]
            if candidates:
                # sort by numeric suffix if possible
                def keyfn(p):
                    try:
                        return int(p.name.split('-')[-1])
                    except Exception:
                        return p.stat().st_mtime_ns
                chosen = sorted(candidates, key=keyfn, reverse=True)[0]
                checkpoint_path = chosen
            else:
                print('No checkpoint directories found under', ck)
                return
        else:
            print('Checkpoint path is empty:', ck)
            return
    else:
        checkpoint_path = ck

    print('Using checkpoint:', checkpoint_path)

    # Find label mapping
    ds_path = project_root / args.dataset
    label_map_file = find_label_mapping(ds_path, checkpoint_path)
    if not label_map_file:
        print('No label_mapping.json found in dataset or checkpoint. Please provide one.')
        return

    with open(label_map_file, 'r', encoding='utf-8') as f:
        lm = json.load(f)

    label2id = lm.get('label2id') or lm.get('id2label')
    id2label = lm.get('id2label') or lm.get('label2id')
    # Ensure id2label keys are ints
    id2label = {int(k): v for k, v in id2label.items()}

    # Load tokenizer and model from checkpoint (tokenizer may be missing in checkpoint)
    print('Loading tokenizer and model from checkpoint...')
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path), local_files_only=True, use_fast=True)
    except Exception:
        print('Tokenizer not found in checkpoint or failed to load; falling back to base model tokenizer')
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, use_fast=True)

    try:
        model = AutoModelForTokenClassification.from_pretrained(str(checkpoint_path), local_files_only=True)
    except Exception:
        print('Model weights not found in checkpoint or failed to load from checkpoint path')
        raise

    # Initialize FinancialNERModel and attach
    fm = FinancialNERModel(cfg, label2id=label2id, id2label=id2label)
    fm.tokenizer = tokenizer
    fm.model = model

    # Save to output
    out_path = project_root / args.out
    fm.save_model(out_path)
    print('Saved model to', out_path)


if __name__ == '__main__':
    main()
