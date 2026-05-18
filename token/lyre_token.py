import os
import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# --- Configuration des chemins dans /tmp ---
""" à changer selon votre ordinateur """
JSONL_FILES = ["/tmp/data/wikipedia_fr.jsonl", "/tmp/data/culturax_fr.jsonl"]
TXT_FILES = ["/tmp/data/wiki_fr_raw.txt", "/tmp/data/culturax_fr_raw.txt"]
OUTPUT_TOKENIZER = "models/tokenizer_lyre.json"

#si vous avez déjà des fichiers textes vous pouvez supprimer ça 
def jsonl_to_txt(jsonl_path, txt_path):
    #test existance
    if not os.path.exists(jsonl_path):
        print(f"Erreur : {jsonl_path} introuvable.")
        return False
        
    print(f"Extraction de {jsonl_path} vers {txt_path}...")
    #supression du formatage des données que j'avais fait. il y a fort à parier que vous n'aurez pas le même problème que moi ici
    with open(jsonl_path, 'r', encoding='utf-8') as infile, open(txt_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    context = data.get("context", "")
                    if context:
                        outfile.write(context.strip() + "\n")
                except json.JSONDecodeError:
                    continue
    return True

# véritable tokenization par BPE on prend les parties qui apparaissent le plus
def train_lyre_tokenizer(files, vocab_size=50257):
    """Entraîne le tokenizer BPE sur les fichiers """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        show_progress=True
    )

    print(f"Entraînement du tokenizer sur {files}...")
    tokenizer.train(files=files, trainer=trainer)

    # Sauvegarde finale
    os.makedirs(os.path.dirname(OUTPUT_TOKENIZER), exist_ok=True)
    tokenizer.save(OUTPUT_TOKENIZER)
    print(f"Tokenizer sauvegardé avec succès dans {OUTPUT_TOKENIZER}")

if __name__ == "__main__":
    # 1. Extraction en texte brut
    extracted_files = []
    for jsonl, txt in zip(JSONL_FILES, TXT_FILES):
        if jsonl_to_txt(jsonl, txt):
            extracted_files.append(txt)
    
    # 2. Entraînement si des fichiers ont été générés
    if extracted_files:
        # le vocab va s'adapter selon ce qu'il arrive à merge si français/(langue proche) ça devrait tourner autour des 30k
        train_lyre_tokenizer(extracted_files, vocab_size=50257)
    else:
        print("Aucun fichier disponible pour l'entraînement.")