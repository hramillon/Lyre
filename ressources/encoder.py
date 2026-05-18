import os
import numpy as np
from tqdm import tqdm
from tokenizers import Tokenizer

# --- Configuration des chemins ---
TOKENIZER_PATH = "models/tokenizer_lyre.json"
INPUT_FILES = ["/tmp/data/wiki_fr_raw.txt", "/tmp/data/culturax_fr_raw.txt"]
TMP_OUTPUT = "/tmp/corpus_encoded_2.bin"
FINAL_OUTPUT = "ressources/corpus_encoded_2.bin"

def encode_to_disk(files):
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    sep_id = tokenizer.token_to_id("[SEP]")
    
    # Écriture directe dans /tmp pour ne pas mettre sur réseau
    with open(TMP_OUTPUT, 'wb') as f_out:
        for file_path in files:
            if not os.path.exists(file_path):
                print(f"Fichier introuvable : {file_path}, skip.")
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f_in:
                for line in tqdm(f_in, desc=f"Encoding {file_path}"):
                    line = line.strip()
                    if line:
                        ids = tokenizer.encode(line).ids
                        ids.append(sep_id)
                        data = np.array(ids, dtype=np.uint32).tobytes()
                        f_out.write(data)

if __name__ == "__main__":
    if os.path.exists(TMP_OUTPUT):
        os.remove(TMP_OUTPUT)
        
    # Encodage
    encode_to_disk(INPUT_FILES)
    
    # Déplacement du fichier final de /tmp vers le workspace
    if os.path.exists(TMP_OUTPUT):
        print(f"Déplacement de {TMP_OUTPUT} vers {FINAL_OUTPUT}...")
        os.rename(TMP_OUTPUT, FINAL_OUTPUT)
        print("Opération terminée avec succès.")