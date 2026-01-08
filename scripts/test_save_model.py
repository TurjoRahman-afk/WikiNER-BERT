from types import SimpleNamespace
from pathlib import Path
from src.models.ner_model import FinancialNERModel

# Create minimal config
cfg = SimpleNamespace()
cfg.model = SimpleNamespace(name='bert-base-uncased', max_length=128, num_labels=2)
cfg.paths = SimpleNamespace(model_dir='./models')

m = FinancialNERModel(cfg)
# Mock model and tokenizer with save_pretrained
class Mock:
    def save_pretrained(self, path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / 'mock.txt').write_text('ok')

m.model = Mock()
m.tokenizer = Mock()
# Call save_model with string
m.save_model('models/test_save')
print('save_model test OK')
