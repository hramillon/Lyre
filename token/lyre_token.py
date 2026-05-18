import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

TXT_FILES = ["/tmp/data/wiki_fr_raw.txt", "/tmp/data/culturax_fr_raw.txt"]
OUTPUT_TOKENIZER = "token/tokenizer_lyre.json"

def train_lyre_tokenizer(files):
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("Aucun fichier txt trouvé !")
        return

    print(f"Fichiers utilisés : {files}")
    for f in files:
        size = os.path.getsize(f) / 1024**3
        print(f"  {f} : {size:.2f} Go")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(
        vocab_size=32768, 
        min_frequency=2,   
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        show_progress=True
    )

    print("Entraînement dynamique du tokenizer...")
    tokenizer.train(files=files, trainer=trainer)

    os.makedirs(os.path.dirname(OUTPUT_TOKENIZER), exist_ok=True)
    tokenizer.save(OUTPUT_TOKENIZER)

    real_vocab = len(tokenizer.get_vocab())
    print(f"Tokenizer sauvegardé : {OUTPUT_TOKENIZER}")
    print(f"Vocab réel à mettre dans le LLM : {real_vocab}")

if __name__ == "__main__":
    train_lyre_tokenizer(TXT_FILES)